// swamp-gpu/kernels/fused_attention.cu
// GPU-accelerated fused attention for LLamanyon.rs
// Target: sm_75 (GTX 1650 Mobile, Turing, no Tensor Cores)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Templated attention kernels: T_KV = float or half for KV cache
// ---------------------------------------------------------------------------

// Kernel 1: compute QK^T scores with scaling
// Each thread computes score[h, t] = Q[h, :] @ K[kv_h, t, :] * scale
// Grid: (n_heads, seq_len)
// Block: 128 threads, each thread handles one (head, t) pair
template<typename T_KV>
__global__ void kernel_scores(
    const float* __restrict__ q,        // [n_heads, head_dim]
    const T_KV* __restrict__ k_cache,   // [n_kv_heads, k_stride, head_dim]
    float* __restrict__ scores,         // [n_heads, seq_len]
    int n_heads,
    int n_kv_heads,
    int seq_len,
    int head_dim,
    int k_stride,
    float scale
) {
    int h = blockIdx.x;   // query head
    int t = blockIdx.y;   // key position

    if (h >= n_heads || t >= seq_len) return;

    int kv_h = h * n_kv_heads / n_heads;

    const float* q_row  = q + h * head_dim;
    const T_KV* k_row  = k_cache + ((size_t)kv_h * k_stride + t) * head_dim;

    float dot = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        dot += q_row[d] * (float)k_row[d];
    }

    // Warp reduce
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        dot += __shfl_xor_sync(0xffffffff, dot, offset);
    }

    // Block reduce
    __shared__ float shared[32]; // one per warp
    int warp_id = threadIdx.x / warpSize;
    int lane   = threadIdx.x % warpSize;
    if (lane == 0) shared[warp_id] = dot;
    __syncthreads();

    if (warp_id == 0) {
        dot = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            dot += __shfl_xor_sync(0xffffffff, dot, offset);
        }
        if (threadIdx.x == 0) {
            scores[h * seq_len + t] = dot * scale;
        }
    }
}

// Kernel 2: fused softmax + weighted sum of V
// Each block handles one head
// Threads: head_dim per block (64), each thread computes one output element
template<typename T_KV>
__global__ void kernel_softmax_weighted_sum(
    const float* __restrict__ scores,   // [n_heads, seq_len]
    const T_KV* __restrict__ v_cache,   // [n_kv_heads, v_stride, head_dim]
    float* __restrict__ output,         // [n_heads, head_dim]
    int n_heads,
    int n_kv_heads,
    int seq_len,
    int head_dim,
    int v_stride
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;

    int kv_h = h * n_kv_heads / n_heads;
    int d = threadIdx.x;

    // Shared memory for scores: seq_len floats (up to 2048 = 8KB, fits in 48KB shared)
    extern __shared__ float sh_scores[];

    // Cooperative group for this block
    auto g = cg::this_thread_block();

    // Each thread loads multiple score values
    for (int t = d; t < seq_len; t += blockDim.x) {
        sh_scores[t] = scores[h * seq_len + t];
    }
    g.sync();

    // Online softmax
    float max_val = -1e10f;
    for (int t = 0; t < seq_len; t++) {
        max_val = fmaxf(max_val, sh_scores[t]);
    }

    float sum_exp = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        sum_exp += __expf(sh_scores[t] - max_val);
    }
    float inv_sum = 1.0f / sum_exp;

    // Weighted sum of V
    float acc = 0.0f;
    const T_KV* v_base = v_cache + (size_t)kv_h * v_stride * head_dim;

    for (int t = 0; t < seq_len; t++) {
        float prob = __expf(sh_scores[t] - max_val) * inv_sum;
        acc += prob * (float)v_base[t * head_dim + d];
    }

    output[h * head_dim + d] = acc;
}

// ---------------------------------------------------------------------------
// Host-callable C API
// ---------------------------------------------------------------------------

extern "C" {

// Initialize CUDA context on device 0
int gpu_init() {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_init: cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Pre-warm by creating a context
    cudaFree(0);
    return 0;
}

// Allocate device memory
void* gpu_alloc(size_t bytes) {
    void* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_alloc(%zu) failed: %s\n", bytes, cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
}

// Free device memory
void gpu_free(void* ptr) {
    cudaFree(ptr);
}

// Copy from host to device
int gpu_copy_to_device(void* dst, const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_copy_to_device failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Copy from device to host
int gpu_copy_to_host(void* dst, const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_copy_to_host failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Synchronize device
int gpu_sync() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sync failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Device-pointer attention: assumes all pointers are already on GPU.
// Only allocates internal temp buffer for scores.
// kv_stride: striding between kv_head blocks (seq_len for contiguous, max_seq_len for persistent)
int gpu_attention_device(
    const float* d_q,         // device: [n_heads, head_dim]
    const float* d_k_cache,   // device: [n_kv_heads, kv_stride, head_dim]
    const float* d_v_cache,   // device: [n_kv_heads, kv_stride, head_dim]
    float* d_output,          // device: [n_heads, head_dim]
    int n_heads,
    int n_kv_heads,
    int seq_len,
    int head_dim,
    int kv_stride
) {
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t scores_size = n_heads * seq_len * sizeof(float);

    float *d_scores = NULL;
    cudaError_t err = cudaMalloc(&d_scores, scores_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_attention_device: cudaMalloc scores failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Kernel 1: scores
    dim3 grid_scores(n_heads, seq_len);
    kernel_scores<float><<<grid_scores, 128>>>(d_q, d_k_cache, d_scores,
                                        n_heads, n_kv_heads, seq_len, head_dim, kv_stride, scale);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel_scores launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_scores);
        return -1;
    }

    // Kernel 2: softmax + weighted sum
    int shared_mem_size = seq_len * sizeof(float);
    kernel_softmax_weighted_sum<float><<<n_heads, head_dim, shared_mem_size>>>(
        d_scores, d_v_cache, d_output,
        n_heads, n_kv_heads, seq_len, head_dim, kv_stride
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel_softmax_weighted_sum launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_scores);
        return -1;
    }

    cudaFree(d_scores);
    return 0;
}

// Allocate persistent KV buffers on GPU: [n_kv_heads, max_seq_len, head_dim]
// Returns device pointer, writes size to *out_bytes
void* gpu_alloc_kv_buffer(int n_kv_heads, int max_seq_len, int head_dim, size_t* out_bytes) {
    size_t bytes = (size_t)n_kv_heads * max_seq_len * head_dim * sizeof(float);
    void* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_alloc_kv_buffer failed: %s\n", cudaGetErrorString(err));
        if (out_bytes) *out_bytes = 0;
        return NULL;
    }
    if (out_bytes) *out_bytes = bytes;
    return ptr;
}

// Copy a single (kv_head, pos) K/V entry from host to GPU buffer
// d_buf: device buffer [n_kv_heads, max_seq_len, head_dim]
// h_src: host source [num_kv_heads, head_dim] (one position's worth, but only head_dim elements used)
int gpu_copy_kv_to_buffer(
    float* d_buf,
    const float* h_src,
    int kv_head,
    int pos,
    int n_kv_heads,
    int max_seq_len,
    int head_dim
) {
    size_t offset = ((size_t)kv_head * max_seq_len + pos) * head_dim;
    cudaError_t err = cudaMemcpy(
        d_buf + offset,
        h_src,
        (size_t)head_dim * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_copy_kv_to_buffer failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Copy an entire layer position (all kv_heads) from host to GPU buffer in one call
// h_src: host source [n_kv_heads, head_dim] contiguous
int gpu_copy_kv_layer(
    float* d_buf,
    const float* h_src,
    int pos,
    int n_kv_heads,
    int max_seq_len,
    int head_dim
) {
    for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
        size_t offset = ((size_t)kv_h * max_seq_len + pos) * head_dim;
        cudaError_t err = cudaMemcpy(
            d_buf + offset,
            h_src + kv_h * head_dim,
            (size_t)head_dim * sizeof(float),
            cudaMemcpyHostToDevice
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu_copy_kv_layer (kv_h=%d) failed: %s\n", kv_h, cudaGetErrorString(err));
            return -1;
        }
    }
    return 0;
}

// Copy Q from host to GPU (small, can be done per layer)
void* gpu_alloc_and_copy_q(const float* h_q, int n_heads, int head_dim) {
    size_t bytes = (size_t)n_heads * head_dim * sizeof(float);
    void* d_q = NULL;
    cudaError_t err = cudaMalloc(&d_q, bytes);
    if (err != cudaSuccess) return NULL;
    err = cudaMemcpy(d_q, h_q, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_q); return NULL; }
    return d_q;
}

// Copy output from GPU to host
int gpu_copy_output_to_host(float* h_out, const float* d_out, int n_heads, int head_dim) {
    size_t bytes = (size_t)n_heads * head_dim * sizeof(float);
    cudaError_t err = cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_copy_output_to_host failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Original host-pointer version (kept for backward compat / testing)
int gpu_attention_forward(
    const float* q,         // [n_heads, head_dim]
    const float* k_cache,   // [n_kv_heads, seq_len, head_dim]
    const float* v_cache,   // [n_kv_heads, seq_len, head_dim]
    float* output,          // [n_heads, head_dim]
    int n_heads,
    int n_kv_heads,
    int seq_len,
    int head_dim
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    size_t q_size      = n_heads * head_dim * sizeof(float);
    size_t kv_size     = n_kv_heads * seq_len * head_dim * sizeof(float);
    size_t scores_size = n_heads * seq_len * sizeof(float);
    size_t out_size    = n_heads * head_dim * sizeof(float);

    float *d_q = NULL, *d_k = NULL, *d_v = NULL;
    float *d_scores = NULL, *d_out = NULL;

    cudaMalloc(&d_q, q_size);
    cudaMalloc(&d_k, kv_size);
    cudaMalloc(&d_v, kv_size);
    cudaMalloc(&d_scores, scores_size);
    cudaMalloc(&d_out, out_size);

    if (!d_q || !d_k || !d_v || !d_scores || !d_out) {
        fprintf(stderr, "gpu_attention_forward: cudaMalloc failed\n");
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
        cudaFree(d_scores); cudaFree(d_out);
        return -1;
    }

    cudaMemcpy(d_q, q, q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k_cache, kv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v_cache, kv_size, cudaMemcpyHostToDevice);

    dim3 grid_scores(n_heads, seq_len);
    kernel_scores<float><<<grid_scores, 128>>>(d_q, d_k, d_scores,
                                        n_heads, n_kv_heads, seq_len, head_dim, seq_len, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel_scores launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
        cudaFree(d_scores); cudaFree(d_out);
        return -1;
    }

    int shared_mem_size = seq_len * sizeof(float);
    kernel_softmax_weighted_sum<float><<<n_heads, head_dim, shared_mem_size>>>(
        d_scores, d_v, d_out,
        n_heads, n_kv_heads, seq_len, head_dim, seq_len
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel_softmax_weighted_sum launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
        cudaFree(d_scores); cudaFree(d_out);
        return -1;
    }

    cudaMemcpy(output, d_out, out_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_scores); cudaFree(d_out);

    return 0;
}

// ---------------------------------------------------------------------------
// CUDA Stream API (for async/overlapped GPU execution)
// ---------------------------------------------------------------------------

cudaStream_t gpu_stream_create() {
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_stream_create failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return stream;
}

void gpu_stream_destroy(cudaStream_t stream) {
    if (stream) cudaStreamDestroy(stream);
}

int gpu_stream_synchronize(cudaStream_t stream) {
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_stream_synchronize failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Async copy: host -> device on given stream
// h_src must be page-locked (pinned) for true async behavior
int gpu_copy_to_device_async(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_copy_to_device_async failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Async copy: device -> host on given stream
int gpu_copy_to_host_async(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_copy_to_host_async failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Async batch KV copy: copy all kv_heads for one position to GPU K/V buffer
int gpu_copy_kv_layer_async(
    float* d_buf,
    const float* h_src,
    int pos,
    int n_kv_heads,
    int max_seq_len,
    int head_dim,
    cudaStream_t stream
) {
    for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
        size_t offset = ((size_t)kv_h * max_seq_len + pos) * head_dim;
        cudaError_t err = cudaMemcpyAsync(
            d_buf + offset,
            h_src + kv_h * head_dim,
            (size_t)head_dim * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu_copy_kv_layer_async (kv_h=%d) failed: %s\n", kv_h, cudaGetErrorString(err));
            return -1;
        }
    }
    return 0;
}

// Stream-based attention: Q already on device, uses given streams
int gpu_attention_streamed(
    const float* d_q,         // device: [n_heads, head_dim]
    const float* d_k_cache,   // device: [n_kv_heads, kv_stride, head_dim]
    const float* d_v_cache,   // device: [n_kv_heads, kv_stride, head_dim]
    float* d_output,          // device: [n_heads, head_dim]
    int n_heads,
    int n_kv_heads,
    int seq_len,
    int head_dim,
    int kv_stride,
    cudaStream_t stream       // compute stream for kernels
) {
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t scores_size = n_heads * seq_len * sizeof(float);

    float *d_scores = NULL;
    cudaError_t err = cudaMalloc(&d_scores, scores_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_attention_streamed: cudaMalloc scores failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    dim3 grid_scores(n_heads, seq_len);
    kernel_scores<float><<<grid_scores, 128, 0, stream>>>(d_q, d_k_cache, d_scores,
                                        n_heads, n_kv_heads, seq_len, head_dim, kv_stride, scale);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel_scores launch (streamed) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_scores);
        return -1;
    }

    int shared_mem_size = seq_len * sizeof(float);
    kernel_softmax_weighted_sum<float><<<n_heads, head_dim, shared_mem_size, stream>>>(
        d_scores, d_v_cache, d_output,
        n_heads, n_kv_heads, seq_len, head_dim, kv_stride
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel_softmax_weighted_sum launch (streamed) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_scores);
        return -1;
    }

    cudaFree(d_scores);
    return 0;
}

// Stream-based attention with FP16 KV cache
int gpu_attention_streamed_half(
    const float* d_q,           // device: [n_heads, head_dim]
    const half* d_k_cache,      // device: [n_kv_heads, kv_stride, head_dim]
    const half* d_v_cache,      // device: [n_kv_heads, kv_stride, head_dim]
    float* d_output,            // device: [n_heads, head_dim]
    int n_heads,
    int n_kv_heads,
    int seq_len,
    int head_dim,
    int kv_stride,
    cudaStream_t stream
) {
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t scores_size = n_heads * seq_len * sizeof(float);

    float *d_scores = NULL;
    cudaError_t err = cudaMalloc(&d_scores, scores_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_attention_streamed_half: cudaMalloc scores failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    dim3 grid_scores(n_heads, seq_len);
    kernel_scores<half><<<grid_scores, 128, 0, stream>>>(d_q, d_k_cache, d_scores,
                                        n_heads, n_kv_heads, seq_len, head_dim, kv_stride, scale);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel_scores<half> launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_scores);
        return -1;
    }

    int shared_mem_size = seq_len * sizeof(float);
    kernel_softmax_weighted_sum<half><<<n_heads, head_dim, shared_mem_size, stream>>>(
        d_scores, d_v_cache, d_output,
        n_heads, n_kv_heads, seq_len, head_dim, kv_stride
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel_softmax_weighted_sum<half> launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_scores);
        return -1;
    }

    cudaFree(d_scores);
    return 0;
}

// ---------------------------------------------------------------------------
// CUDA Graph API: capture attention compute into a reusable graph
// Eliminates kernel launch overhead for repeated attention calls
// ---------------------------------------------------------------------------

// Create a CUDA Graph executable for the attention compute pipeline.
// All device pointers are fixed (pre-allocated buffers).
// Uses max_seq throughout; actual seq_len is set via set_params before each replay.
// Returns opaque handle, writes output parameters.

// Create a CUDA Graph executable for attention compute.
// All parameters (including seq_len) are FIXED at graph creation time.
// Caller should cache graphs by seq_len and create one per distinct value.
void* gpu_graph_create_attention(
    const float* d_q, const float* d_k_cache, const float* d_v_cache,
    float* d_scores, float* d_output,
    int n_heads, int n_kv_heads, int seq_len, int head_dim, int kv_stride
) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    float scale = 1.0f / sqrtf((float)head_dim);
    dim3 grid_scores(n_heads, seq_len);
    int shared_mem_softmax = seq_len * sizeof(float);

    // Capture kernel launches into the graph
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    kernel_scores<float><<<grid_scores, 128, 0, stream>>>(
        d_q, d_k_cache, d_scores,
        n_heads, n_kv_heads, seq_len, head_dim, kv_stride, scale
    );

    kernel_softmax_weighted_sum<float><<<n_heads, head_dim, shared_mem_softmax, stream>>>(
        d_scores, d_v_cache, d_output,
        n_heads, n_kv_heads, seq_len, head_dim, kv_stride
    );

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_graph_create: cudaStreamEndCapture failed: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        return NULL;
    }

    // Instantiate the executable graph
    cudaGraphExec_t graph_exec;
    err = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_graph_create: cudaGraphInstantiate failed: %s\n", cudaGetErrorString(err));
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        return NULL;
    }

    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return (void*)graph_exec;
}

// Replay a fixed-parameter attention graph.
// All parameters must match the graph creation parameters exactly.
// Returns 0 on success, -1 on error.
int gpu_graph_replay_attention(
    void* graph_handle,
    cudaStream_t stream
) {
    cudaGraphExec_t graph_exec = (cudaGraphExec_t)graph_handle;
    cudaError_t err = cudaGraphLaunch(graph_exec, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_graph_replay: cudaGraphLaunch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Destroy a CUDA Graph executable
void gpu_graph_destroy(void* graph_handle) {
    if (graph_handle) {
        cudaGraphExecDestroy((cudaGraphExec_t)graph_handle);
    }
}

// Create a CUDA Graph executable for attention with FP16 KV cache.
void* gpu_graph_create_attention_half(
    const float* d_q, const half* d_k_cache, const half* d_v_cache,
    float* d_scores, float* d_output,
    int n_heads, int n_kv_heads, int seq_len, int head_dim, int kv_stride
) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    float scale = 1.0f / sqrtf((float)head_dim);
    dim3 grid_scores(n_heads, seq_len);
    int shared_mem_softmax = seq_len * sizeof(float);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    kernel_scores<half><<<grid_scores, 128, 0, stream>>>(
        d_q, d_k_cache, d_scores,
        n_heads, n_kv_heads, seq_len, head_dim, kv_stride, scale
    );

    kernel_softmax_weighted_sum<half><<<n_heads, head_dim, shared_mem_softmax, stream>>>(
        d_scores, d_v_cache, d_output,
        n_heads, n_kv_heads, seq_len, head_dim, kv_stride
    );

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_graph_create_half: cudaStreamEndCapture failed: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        return NULL;
    }

    cudaGraphExec_t graph_exec;
    err = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_graph_create_half: cudaGraphInstantiate failed: %s\n", cudaGetErrorString(err));
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        return NULL;
    }

    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return (void*)graph_exec;
}

// ---------------------------------------------------------------------------
// FP16 KV cache support
// ---------------------------------------------------------------------------

// Strided scatter kernel: copy contiguous float → strided half
// src: contiguous [n_kv_heads * head_dim] floats (GPU staging)
// dst: [n_kv_heads, max_seq_len, head_dim] half, writes at position `pos` for each kv_head
__global__ void copy_float_to_half_strided(
    const float* __restrict__ src,
    half* __restrict__ dst,
    int pos,
    int n_kv_heads,
    int max_seq_len,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_kv_heads * head_dim;
    if (idx >= total) return;
    int kv_h = idx / head_dim;
    int d = idx % head_dim;
    size_t dst_offset = ((size_t)kv_h * max_seq_len + pos) * head_dim + d;
    dst[dst_offset] = __float2half(src[idx]);
}

// Pre-allocated staging buffer for float→half conversion (lazily allocated)
static float* g_half_staging = NULL;
static size_t g_half_staging_size = 0;

// Allocate persistent half-precision KV buffer: [n_kv_heads, max_seq_len, head_dim]
void* gpu_alloc_kv_buffer_half(int n_kv_heads, int max_seq_len, int head_dim, size_t* out_bytes) {
    size_t num_elements = (size_t)n_kv_heads * max_seq_len * head_dim;
    size_t bytes = num_elements * sizeof(half);
    void* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_alloc_kv_buffer_half failed: %s\n", cudaGetErrorString(err));
        if (out_bytes) *out_bytes = 0;
        return NULL;
    }
    if (out_bytes) *out_bytes = bytes;
    return ptr;
}

// Async copy float host data → half GPU buffer for one layer position (all kv_heads).
// Uses a reusable GPU staging buffer for float→half conversion.
int gpu_copy_kv_layer_async_half(
    half* d_buf,
    const float* h_src,
    int pos,
    int n_kv_heads,
    int max_seq_len,
    int head_dim,
    cudaStream_t stream
) {
    size_t layer_elements = (size_t)n_kv_heads * head_dim;
    size_t layer_bytes_float = layer_elements * sizeof(float);

    // Ensure staging buffer is large enough
    if (g_half_staging == NULL || g_half_staging_size < layer_bytes_float) {
        if (g_half_staging) cudaFree(g_half_staging);
        cudaError_t err = cudaMalloc(&g_half_staging, layer_bytes_float);
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu_copy_kv_layer_async_half: cudaMalloc staging failed: %s\n", cudaGetErrorString(err));
            g_half_staging = NULL;
            g_half_staging_size = 0;
            return -1;
        }
        g_half_staging_size = layer_bytes_float;
    }

    // Copy host float → device staging
    cudaError_t err = cudaMemcpyAsync(
        g_half_staging, h_src, layer_bytes_float,
        cudaMemcpyHostToDevice, stream
    );
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_copy_kv_layer_async_half: cudaMemcpyAsync failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Launch strided scatter kernel: staging (float) → d_buf at position (half)
    dim3 conv_grid((layer_elements + 255) / 256);
    copy_float_to_half_strided<<<conv_grid, 256, 0, stream>>>(
        g_half_staging, d_buf, pos, n_kv_heads, max_seq_len, head_dim
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_copy_kv_layer_async_half: strided scatter failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}


// CUDA event API for HLC (Hybrid Logical Clock) correlation
cudaEvent_t gpu_event_create() {
    cudaEvent_t event;
    cudaError_t err = cudaEventCreate(&event);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_event_create failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return event;
}

int gpu_event_record(cudaEvent_t event, cudaStream_t stream) {
    cudaError_t err = cudaEventRecord(event, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_event_record failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int gpu_event_synchronize(cudaEvent_t event) {
    cudaError_t err = cudaEventSynchronize(event);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_event_synchronize failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

float gpu_event_elapsed_ms(cudaEvent_t start, cudaEvent_t end) {
    float ms = 0.0f;
    cudaError_t err = cudaEventElapsedTime(&ms, start, end);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_event_elapsed_ms failed: %s\n", cudaGetErrorString(err));
        return -1.0f;
    }
    return ms;
}

void gpu_event_destroy(cudaEvent_t event) {
    cudaEventDestroy(event);
}


// ---------------------------------------------------------------------------
// Q4_K GEMV: dequantize on-the-fly, VNNI-style dot product
// Each block: 256 threads, each processes one output row
// W: [n_rows, n_blocks * 144] Q4_K in VRAM
// x: [n_cols] f32 activation
// out: [n_rows] f32 output
// ---------------------------------------------------------------------------
__global__ void kernel_gemv_q4k(
    const uint8_t* __restrict__ w,       // Q4_K weights in VRAM
    const float*   __restrict__ x,       // activation vector
    float*         __restrict__ out,     // output vector
    int n_rows,
    int n_blocks,
    float scale_x,
    float inv_scale_x
) {
    // x is pre-quantized to i8 on host, stored as f32 for simplicity
    // Each thread processes 1 row
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    float total = 0.0f;
    for (int blk = 0; blk < n_blocks; blk++) {
        const uint8_t* blk_ptr = w + ((size_t)row * n_blocks + blk) * 144;
        
        // Load d/dmin as f16
        half d_h = *reinterpret_cast<const half*>(blk_ptr);
        half dmin_h = *reinterpret_cast<const half*>(blk_ptr + 2);
        float d = __half2float(d_h);
        float dmin = __half2float(dmin_h);
        
        // Unpack scales (12 bytes → 8+8)
        float scales[8], mins[8];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            scales[j] = (blk_ptr[4 + j] & 63);
            mins[j]   = (blk_ptr[8 + j] & 63);
        }
        #pragma unroll
        for (int j = 4; j < 8; j++) {
            scales[j] = (blk_ptr[8 + j] & 0xF) | ((blk_ptr[4 + j - 4] >> 6) << 4);
            mins[j]   = (blk_ptr[8 + j] >> 4) | ((blk_ptr[4 + j] >> 6) << 4);
        }
        
        // Process 8 sub-blocks of 32 values each
        const uint8_t* qs = blk_ptr + 16;
        float dot = 0.0f;
        float dot_corr = 0.0f;
        
        for (int sb = 0; sb < 8; sb++) {
            float sum_x = 0.0f;
            #pragma unroll
            for (int k = 0; k < 32; k++) {
                int nib = (qs[sb * 16 + k / 2] >> ((k % 2) * 4)) & 0x0F;
                float w_val = d * (nib - 8) * scales[sb] + dmin * mins[sb];
                float xk = x[blk * 256 + sb * 32 + k];
                dot += w_val * xk;
            }
        }
        total += dot;
    }
    out[row] = total;
}

void gpu_gemv_q4k(
    const uint8_t* d_w, const float* d_x, float* d_out,
    int n_rows, int n_blocks,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n_rows + threads - 1) / threads;
    kernel_gemv_q4k<<<blocks, threads, 0, stream>>>(d_w, d_x, d_out, n_rows, n_blocks, 0.0f, 0.0f);
}

// Upload weights to GPU (ring buffer)
void gpu_upload_weights(const uint8_t* h_w, uint8_t** d_w, size_t bytes, cudaStream_t stream) {
    cudaMalloc(d_w, bytes);
    cudaMemcpyAsync(*d_w, h_w, bytes, cudaMemcpyHostToDevice, stream);
}

void gpu_free_weights(uint8_t* d_w) {
    cudaFree(d_w);
}

// Copy activation vector to GPU and run GEMV, copy result back
void gpu_gemv_q4k_full(
    const uint8_t* d_w, const float* h_x, float* h_out,
    int n_rows, int n_blocks, cudaStream_t stream
) {
    size_t x_bytes = (size_t)n_blocks * 256 * sizeof(float);
    size_t out_bytes = (size_t)n_rows * sizeof(float);
    float *d_x, *d_out;
    cudaMalloc(&d_x, x_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpyAsync(d_x, h_x, x_bytes, cudaMemcpyHostToDevice, stream);
    int threads = 256;
    int blocks = (n_rows + threads - 1) / threads;
    kernel_gemv_q4k<<<blocks, threads, 0, stream>>>(d_w, d_x, d_out, n_rows, n_blocks, 0.0f, 0.0f);
    cudaMemcpyAsync(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost, stream);
    cudaFree(d_x);
    cudaFree(d_out);
}

// GEMV with pre-allocated buffers (no malloc per call)
void gpu_gemv_q4k_prealloc(
    const uint8_t* d_w, const float* h_x, float* h_out,
    float* d_x, float* d_out,
    int n_rows, int n_blocks, int max_n_rows, int max_n_cols,
    cudaStream_t stream
) {
    size_t x_bytes = (size_t)n_blocks * 256 * sizeof(float);
    size_t out_bytes = (size_t)n_rows * sizeof(float);
    cudaMemcpyAsync(d_x, h_x, x_bytes, cudaMemcpyHostToDevice, stream);
    int threads = 256;
    int blocks = (n_rows + threads - 1) / threads;
    kernel_gemv_q4k<<<blocks, threads, 0, stream>>>(d_w, d_x, d_out, n_rows, n_blocks, 0.0f, 0.0f);
    cudaMemcpyAsync(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost, stream);
}

void gpu_alloc_buffers(float** d_x, float** d_out, int max_cols, int max_rows, cudaStream_t stream) {
    cudaMalloc(d_x, (size_t)max_cols * sizeof(float));
    cudaMalloc(d_out, (size_t)max_rows * sizeof(float));
}

void gpu_free_buffers(float* d_x, float* d_out) {
    cudaFree(d_x);
    cudaFree(d_out);
}
// ===========================================================================
// Swamp Continuum: meta-kernel CUDA persistente
// Lê opcodes de um ring buffer em device memory. Nunca retorna.
// CPU publica opcodes de 32 bytes. GPU interpreta e executa.
// ===========================================================================

struct __align__(32) SwampOpcode {
    uint8_t  op;             // 0=GEMV_Q4K, 1=ATTN_SPARSE, 2=FFN_SILU_MUL
    uint8_t  flags;
    uint16_t layer_id;
    uint32_t x_offset;       // offset no buffer persistente de input
    uint32_t w_offset;       // offset nos pesos Q4_K
    uint32_t out_offset;     // offset no buffer persistente de output
    uint16_t rows;
    uint16_t cols;
    uint16_t n_blocks;
    uint16_t head_dim;
    uint8_t  reserved[6];
};

// Ring buffer produtor-consumidor (GPU lê, CPU escreve)
struct SwampRingBuffer {
    volatile uint32_t head;   // GPU consumiu até aqui
    volatile uint32_t tail;   // CPU escreveu até aqui
    SwampOpcode slots[1024];  // opcodes circulares
};

// Buffer persistente de estados (x, out) — GPU mantém entre opcodes
#define MAX_STATE_SIZE (4 * 1024 * 1024) // 4MB de estados em VRAM

// ---------------------------------------------------------------------------
// Processa um opcode GEMV_Q4K
// ---------------------------------------------------------------------------
__device__ void exec_gemv_q4k(
    const SwampOpcode* op,
    const uint8_t* d_w_base,
    float* d_state
) {
    const uint8_t* w = d_w_base + op->w_offset;
    float* x = d_state + op->x_offset;
    float* out = d_state + op->out_offset;
    int n_rows = op->rows;
    int n_blocks = op->n_blocks;

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    float total = 0.0f;
    for (int blk = 0; blk < n_blocks; blk++) {
        const uint8_t* blk_ptr = w + ((size_t)row * n_blocks + blk) * 144;
        half d_h = *reinterpret_cast<const half*>(blk_ptr);
        half dmin_h = *reinterpret_cast<const half*>(blk_ptr + 2);
        float d = __half2float(d_h);
        float dmin = __half2float(dmin_h);

        float scales[8], mins[8];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            scales[j] = (blk_ptr[4 + j] & 63);
            mins[j]   = (blk_ptr[8 + j] & 63);
        }
        #pragma unroll
        for (int j = 4; j < 8; j++) {
            scales[j] = (blk_ptr[8 + j] & 0xF) | ((blk_ptr[4 + j - 4] >> 6) << 4);
            mins[j]   = (blk_ptr[8 + j] >> 4) | ((blk_ptr[4 + j] >> 6) << 4);
        }

        const uint8_t* qs = blk_ptr + 16;
        float dot = 0.0f;
        for (int sb = 0; sb < 8; sb++) {
            for (int k = 0; k < 32; k++) {
                int nib = (qs[sb * 16 + k / 2] >> ((k % 2) * 4)) & 0x0F;
                float w_val = d * (nib - 8) * scales[sb] + dmin * mins[sb];
                dot += w_val * x[blk * 256 + sb * 32 + k];
            }
        }
        total += dot;
    }
    out[row] = total;
}

// ---------------------------------------------------------------------------
// Persistent kernel: processa opcodes em loop infinito
// ---------------------------------------------------------------------------
__global__ void swamp_continuum(
    SwampRingBuffer* ring,
    const uint8_t* d_w_base,
    float* d_state,
    volatile int* shutdown_flag
) {
    // Shared memory: thread 0 publica o opcode atual, todos os threads executam
    __shared__ SwampOpcode shared_op;
    __shared__ volatile int op_ready;

    while (true) {
        if (shutdown_flag && *shutdown_flag) return;

        // Thread 0: gerencia ring buffer
        if (threadIdx.x == 0) {
            __threadfence_system();
            uint32_t tail = ring->tail;
            uint32_t head = ring->head;

            if (head != tail) {
                uint32_t slot = head & 1023;
                shared_op = ring->slots[slot];
                op_ready = 1;
            } else {
                op_ready = 0;
            }
        }

        __syncthreads();

        if (!op_ready) continue;

        // Todos os threads executam o GEMV
        if (shared_op.op == 0) {
            exec_gemv_q4k(&shared_op, d_w_base, d_state);
        }

        __syncthreads();

        // Thread 0: avança head
        if (threadIdx.x == 0) {
            if (shared_op.op == 1) return; // SHUTDOWN
            __threadfence_system();
            ring->head = ring->head + 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Host-callable wrappers
// ---------------------------------------------------------------------------

void gpu_swamp_init(
    SwampRingBuffer** d_ring,
    float** d_state,
    int** d_shutdown,
    cudaStream_t stream
) {
    cudaMalloc((void**)d_ring, sizeof(SwampRingBuffer));
    cudaMemset((void*)*d_ring, 0, sizeof(SwampRingBuffer));
    cudaMalloc((void**)d_state, MAX_STATE_SIZE);
    cudaMemset((void*)*d_state, 0, MAX_STATE_SIZE);
    cudaMalloc((void**)d_shutdown, sizeof(int));
    cudaMemset((void*)*d_shutdown, 0, sizeof(int));
}

// Buffer pinned para opcodes (alocado uma vez, reutilizado)
static SwampOpcode* pinned_op_buf = NULL;

void gpu_swamp_alloc_pinned() {
    if (pinned_op_buf == NULL) {
        cudaHostAlloc(&pinned_op_buf, sizeof(SwampOpcode), cudaHostAllocDefault);
    }
}

void gpu_swamp_launch(
    SwampRingBuffer* d_ring,
    const uint8_t* d_w_base,
    float* d_state,
    int* d_shutdown,
    cudaStream_t kernel_stream  // stream DEDICADO (nunca sincronizado)
) {
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    if (sm_count < 1) sm_count = 1;
    swamp_continuum<<<sm_count, 256, 0, kernel_stream>>>(d_ring, d_w_base, d_state, d_shutdown);
    gpu_swamp_alloc_pinned();
}

void gpu_swamp_free_pinned() {
    if (pinned_op_buf) cudaFreeHost(pinned_op_buf);
}

// Enfileira opcode via stream de dados (NÃO o stream do kernel)
void gpu_swamp_enqueue(
    SwampRingBuffer* d_ring,
    int op_type, int layer_id,
    int x_off, int w_off, int out_off,
    int rows, int n_blocks,
    unsigned int local_tail,
    cudaStream_t data_stream
) {
    if (!pinned_op_buf) return;

    // Preenche opcode no buffer pinned
    pinned_op_buf->op = op_type;
    pinned_op_buf->layer_id = layer_id;
    pinned_op_buf->x_offset = x_off;
    pinned_op_buf->w_offset = w_off;
    pinned_op_buf->out_offset = out_off;
    pinned_op_buf->rows = rows;
    pinned_op_buf->n_blocks = n_blocks;

    // Copia opcode para o ring buffer no device (via data_stream)
    cudaMemcpyAsync(
        &d_ring->slots[local_tail & 1023],
        pinned_op_buf, sizeof(SwampOpcode),
        cudaMemcpyHostToDevice,
        data_stream
    );

    // Publica tail (kernel vê via __threadfence_system)
    unsigned int new_tail = local_tail + 1;
    cudaMemcpyAsync(
        (void*)(&d_ring->tail),
        &new_tail, sizeof(unsigned int),
        cudaMemcpyHostToDevice,
        data_stream
    );
}

// Leitura de resultado do device para host (síncrono no data_stream)
void gpu_swamp_readback(float* dst, float* src, size_t bytes, cudaStream_t data_stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, data_stream);
    cudaStreamSynchronize(data_stream);
}

// Sinaliza shutdown do kernel persistente
void gpu_swamp_shutdown(int* d_shutdown, cudaStream_t stream) {
    int val = 1;
    cudaMemcpyAsync(d_shutdown, &val, sizeof(int), cudaMemcpyHostToDevice, stream);
}

// Sync stream (only if not already defined elsewhere)
int gpu_stream_sync(cudaStream_t stream) {
    cudaError_t e = cudaStreamSynchronize(stream);
    return (int)e;
}

// ===========================================================================
// CUDA Graph: GEMV QKV batch — captures copy x + 3 GEMVs + 3 copy backs
// All host pointers must be stable (pinned or fixed Vec), all device ptrs pre-alloc
// ===========================================================================
void* gpu_graph_create_gemv_qkv(
    const uint8_t* d_w_q, const uint8_t* d_w_k, const uint8_t* d_w_v,
    float* d_x, float* d_out,
    const float* h_x, float* h_q, float* h_k, float* h_v,
    int n_rows_q, int n_rows_k, int n_rows_v, int n_blocks,
    int x_bytes, int out_bytes_q, int out_bytes_k, int out_bytes_v
) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    int blocks_q = (n_rows_q + 255) / 256;
    int blocks_k = (n_rows_k + 255) / 256;
    int blocks_v = (n_rows_v + 255) / 256;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    cudaMemcpyAsync(d_x, h_x, x_bytes, cudaMemcpyHostToDevice, stream);
    kernel_gemv_q4k<<<blocks_q, 256, 0, stream>>>(d_w_q, d_x, d_out, n_rows_q, n_blocks, 0.0f, 0.0f);
    cudaMemcpyAsync(h_q, d_out, out_bytes_q, cudaMemcpyDeviceToHost, stream);
    kernel_gemv_q4k<<<blocks_k, 256, 0, stream>>>(d_w_k, d_x, d_out, n_rows_k, n_blocks, 0.0f, 0.0f);
    cudaMemcpyAsync(h_k, d_out, out_bytes_k, cudaMemcpyDeviceToHost, stream);
    kernel_gemv_q4k<<<blocks_v, 256, 0, stream>>>(d_w_v, d_x, d_out, n_rows_v, n_blocks, 0.0f, 0.0f);
    cudaMemcpyAsync(h_v, d_out, out_bytes_v, cudaMemcpyDeviceToHost, stream);

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_graph_create_gemv_qkv: capture failed: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        return NULL;
    }
    cudaGraphExec_t graph_exec;
    err = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_graph_create_gemv_qkv: instantiate failed: %s\n", cudaGetErrorString(err));
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        return NULL;
    }
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return (void*)graph_exec;
}

// CUDA Graph: Gate+Up GEMV batch — copies x once, 2 kernels, 2 copy backs
void* gpu_graph_create_gemv_gate_up(
    const uint8_t* d_w_gate, const uint8_t* d_w_up,
    float* d_x, float* d_out,
    const float* h_x, float* h_gate, float* h_up,
    int n_rows, int n_blocks,
    int x_bytes, int out_bytes
) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    int blocks = (n_rows + 255) / 256;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    cudaMemcpyAsync(d_x, h_x, x_bytes, cudaMemcpyHostToDevice, stream);
    kernel_gemv_q4k<<<blocks, 256, 0, stream>>>(d_w_gate, d_x, d_out, n_rows, n_blocks, 0.0f, 0.0f);
    cudaMemcpyAsync(h_gate, d_out, out_bytes, cudaMemcpyDeviceToHost, stream);
    kernel_gemv_q4k<<<blocks, 256, 0, stream>>>(d_w_up, d_x, d_out, n_rows, n_blocks, 0.0f, 0.0f);
    cudaMemcpyAsync(h_up, d_out, out_bytes, cudaMemcpyDeviceToHost, stream);

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_graph_create_gemv_gate_up: capture failed: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        return NULL;
    }
    cudaGraphExec_t graph_exec;
    err = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        return NULL;
    }
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return (void*)graph_exec;
}

// CUDA Graph: single GEMV — copy x + kernel + copy out
void* gpu_graph_create_gemv_single(
    const uint8_t* d_w,
    float* d_x, float* d_out,
    const float* h_x, float* h_out,
    int n_rows, int n_blocks,
    int x_bytes, int out_bytes
) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    int blocks = (n_rows + 255) / 256;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    cudaMemcpyAsync(d_x, h_x, x_bytes, cudaMemcpyHostToDevice, stream);
    kernel_gemv_q4k<<<blocks, 256, 0, stream>>>(d_w, d_x, d_out, n_rows, n_blocks, 0.0f, 0.0f);
    cudaMemcpyAsync(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost, stream);

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        return NULL;
    }
    cudaGraphExec_t graph_exec;
    err = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        return NULL;
    }
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return (void*)graph_exec;
}

} // extern "C"
