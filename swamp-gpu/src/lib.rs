pub mod vulkan;
pub mod compute_graph;
pub mod shaders;
pub mod mojo;

use std::sync::Arc;
use ash::vk;
use vk::Handle;

pub use vulkan::VkBackend;
pub use compute_graph::{ComputeGraph, ComputeNode, ComputeNodeOp};
pub use shaders::{ShaderCache, ShaderType};
pub use mojo::{MojoKernel, get_mojo};

use swamp_tensors::hma::{DeviceAllocator, Allocation, MemoryLocation};

// ===========================================================================
// New Architecture Types
// ===========================================================================

pub struct GpuDevice {
    pub backend: Arc<VkBackend>,
    pub shader_cache: std::sync::Mutex<ShaderCache>,
    pub compute_graph: std::sync::Mutex<ComputeGraph>,
    pub enabled: bool,
}

impl GpuDevice {
    pub fn new() -> Self {
        let backend = Arc::new(VkBackend::new());
        let enabled = backend.enabled;
        if enabled {
            let sc = ShaderCache::new(&backend);
            match ComputeGraph::new(&backend) {
                Ok(cg) => Self {
                    backend,
                    shader_cache: std::sync::Mutex::new(sc),
                    compute_graph: std::sync::Mutex::new(cg),
                    enabled,
                },
                Err(e) => {
                    tracing::warn!("Failed to create compute graph: {}", e);
                    Self::disabled()
                }
            }
        } else {
            Self::disabled()
        }
    }

    fn disabled() -> Self {
        let backend = Arc::new(VkBackend::new());
        let sc = ShaderCache::new(&backend);
        let cg = ComputeGraph::new(&backend).unwrap_or_else(|_| {
            std::process::abort();
        });
        Self {
            backend,
            shader_cache: std::sync::Mutex::new(sc),
            compute_graph: std::sync::Mutex::new(cg),
            enabled: false,
        }
    }

    pub fn is_operational(&self) -> bool {
        self.enabled
    }

    pub fn submit_graph(&self) -> bool {
        if !self.enabled { return false; }
        let mut cg = self.compute_graph.lock().unwrap();
        cg.submit().is_ok()
    }

    pub fn wait_graph(&self) -> bool {
        if !self.enabled { return false; }
        let cg = self.compute_graph.lock().unwrap();
        cg.wait().is_ok()
    }

    pub fn allocate_device_local(&self, size: u64) -> Option<(vk::Buffer, vk::DeviceMemory)> {
        if !self.enabled { return None; }
        self.backend.allocate_buffer(
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ).ok()
    }

    pub fn allocate_unified(&self, size: u64) -> Option<(vk::Buffer, vk::DeviceMemory)> {
        if !self.enabled { return None; }
        self.backend.allocate_buffer(
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED,
        ).ok()
    }
}

impl DeviceAllocator for GpuDevice {
    fn allocate_device(&self, size: usize) -> Option<Allocation> {
        let (buffer, memory) = self.allocate_device_local(size as u64)?;
        let ptr = if memory != vk::DeviceMemory::null() && size > 0 {
            unsafe {
                self.backend.device.map_memory(
                    memory,
                    0,
                    size as u64,
                    vk::MemoryMapFlags::empty(),
                ).ok()?
            }
        } else {
            std::ptr::null_mut()
        };
        Some(Allocation {
            ptr: ptr as *mut u8,
            size,
            location: MemoryLocation::DeviceLocal,
            device_handle: Some(buffer.as_raw()),
        })
    }

    fn allocate_unified(&self, size: usize) -> Option<Allocation> {
        let (buffer, memory) = self.allocate_unified(size as u64)?;
        let ptr = unsafe {
            self.backend.device.map_memory(
                memory,
                0,
                size as u64,
                vk::MemoryMapFlags::empty(),
            ).ok()?
        };
        Some(Allocation {
            ptr: ptr as *mut u8,
            size,
            location: MemoryLocation::UnifiedMapped,
            device_handle: Some(buffer.as_raw()),
        })
    }

    fn free(&self, alloc: &Allocation) {
        if let Some(handle) = alloc.device_handle {
            let buffer = unsafe { vk::Buffer::from_raw(handle) };
            unsafe { self.backend.device.destroy_buffer(buffer, None); }
        }
    }

    fn upload(&self, dst: &Allocation, src: &[u8]) {
        if !self.enabled || dst.ptr.is_null() { return; }
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.ptr, src.len().min(dst.size)); }
    }

    fn download(&self, dst: &mut [u8], src: &Allocation) {
        if !self.enabled || src.ptr.is_null() { return; }
        unsafe { std::ptr::copy_nonoverlapping(src.ptr, dst.as_mut_ptr(), dst.len().min(src.size)); }
    }
}

pub fn gpu_available() -> bool {
    let backend = VkBackend::new();
    backend.enabled
}

// ===========================================================================
// GpuBuf: thin wrapper around a GPU buffer
// ===========================================================================

#[derive(Clone, Copy)]
pub struct GpuBuf {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    pub mapped: *mut u8,
}

unsafe impl Send for GpuBuf {}
unsafe impl Sync for GpuBuf {}

impl GpuBuf {
    pub fn null() -> Self {
        Self {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 0,
            mapped: std::ptr::null_mut(),
        }
    }

    pub fn is_null(&self) -> bool {
        self.buffer == vk::Buffer::null()
    }

    pub fn as_slice_f32(&self) -> Option<&[f32]> {
        if self.mapped.is_null() { return None; }
        Some(unsafe { std::slice::from_raw_parts(self.mapped as *const f32, self.size as usize / 4) })
    }

    pub fn as_slice_f32_mut(&self) -> Option<&mut [f32]> {
        if self.mapped.is_null() { return None; }
        Some(unsafe { std::slice::from_raw_parts_mut(self.mapped as *mut f32, self.size as usize / 4) })
    }

    pub fn copy_from_host(&self, src: &[u8]) {
        if self.mapped.is_null() || src.is_empty() { return; }
        let len = src.len().min(self.size as usize);
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.mapped, len); }
    }

    pub fn copy_to_host(&self, dst: &mut [u8]) {
        if self.mapped.is_null() || dst.is_empty() { return; }
        let len = dst.len().min(self.size as usize);
        unsafe { std::ptr::copy_nonoverlapping(self.mapped, dst.as_mut_ptr(), len); }
    }
}

// ===========================================================================
// LayerWeightSet: all weight buffers for all layers on device
// ===========================================================================

pub struct LayerWeights {
    pub q: Vec<GpuBuf>,
    pub k: Vec<GpuBuf>,
    pub v: Vec<GpuBuf>,
    pub o: Vec<GpuBuf>,
    pub gate: Vec<GpuBuf>,
    pub up: Vec<GpuBuf>,
    pub down: Vec<GpuBuf>,
    pub attn_norm: Vec<GpuBuf>,
    pub ffn_norm: Vec<GpuBuf>,
}

impl LayerWeights {
    pub fn new(num_layers: usize) -> Self {
        Self {
            q: vec![GpuBuf::null(); num_layers],
            k: vec![GpuBuf::null(); num_layers],
            v: vec![GpuBuf::null(); num_layers],
            o: vec![GpuBuf::null(); num_layers],
            gate: vec![GpuBuf::null(); num_layers],
            up: vec![GpuBuf::null(); num_layers],
            down: vec![GpuBuf::null(); num_layers],
            attn_norm: vec![GpuBuf::null(); num_layers],
            ffn_norm: vec![GpuBuf::null(); num_layers],
        }
    }
}

// ===========================================================================
// GpuKVCache: ring buffer for key/value cache on device
// ===========================================================================

pub struct GpuKVCache {
    pub k_buf: GpuBuf,
    pub v_buf: GpuBuf,
    pub n_kv_heads: usize,
    pub window_size: usize,
    pub head_dim: usize,
}

impl GpuKVCache {
    pub fn new(device: &GpuDevice, n_kv_heads: usize, window_size: usize, head_dim: usize) -> Option<Self> {
        if !device.enabled { return None; }
        let elem_size = 2u64; // FP16
        let buf_size = n_kv_heads as u64 * window_size as u64 * head_dim as u64 * elem_size;
        let (k_buf, k_mem) = device.backend.allocate_buffer(
            buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ).ok()?;
        let (v_buf, v_mem) = device.backend.allocate_buffer(
            buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ).ok()?;
        Some(Self {
            k_buf: GpuBuf { buffer: k_buf, memory: k_mem, size: buf_size, mapped: std::ptr::null_mut() },
            v_buf: GpuBuf { buffer: v_buf, memory: v_mem, size: buf_size, mapped: std::ptr::null_mut() },
            n_kv_heads,
            window_size,
            head_dim,
        })
    }
}

// ===========================================================================
// ScratchBuffers: pre-allocated working buffers for compute graph
// ===========================================================================

pub struct ScratchBuffers {
    pub d_x: GpuBuf,
    pub d_x_norm: GpuBuf,
    pub d_q: GpuBuf,
    pub d_k: GpuBuf,
    pub d_v: GpuBuf,
    pub d_attn: GpuBuf,
    pub d_gate: GpuBuf,
    pub d_up: GpuBuf,
    pub d_down: GpuBuf,
    pub d_scores: GpuBuf,
}

impl ScratchBuffers {
    pub fn new(device: &GpuDevice, embed_dim: usize, ffn_dim: usize, n_heads: usize, window_size: usize, head_dim: usize) -> Option<Self> {
        if !device.enabled { return None; }
        let alloc = |size: usize| -> Option<GpuBuf> {
            let (buf, mem) = device.backend.allocate_buffer(
                size as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ).ok()?;
            Some(GpuBuf { buffer: buf, memory: mem, size: size as u64, mapped: std::ptr::null_mut() })
        };
        let e = embed_dim * 4; // f32 bytes
        let f = ffn_dim * 4;
        let qkv = n_heads.max(1) * head_dim * 4;
        let scores = n_heads * window_size * 4;
        Some(Self {
            d_x: alloc(e)?,
            d_x_norm: alloc(e)?,
            d_q: alloc(qkv)?,
            d_k: alloc(qkv)?,
            d_v: alloc(qkv)?,
            d_attn: alloc(e)?,
            d_gate: alloc(f)?,
            d_up: alloc(f)?,
            d_down: alloc(e)?,
            d_scores: alloc(scores)?,
        })
    }
}

// ===========================================================================
// StagingBuffer: host-visible coherent buffer for CPU↔GPU transfers
// ===========================================================================

pub struct StagingBuffer {
    pub buf: GpuBuf,
    pub device: vk::Buffer,
    pub size: u64,
}

impl StagingBuffer {
    pub fn new(device: &GpuDevice, size: u64) -> Option<Self> {
        if !device.enabled { return None; }
        let (buf, mem) = device.backend.allocate_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ).ok()?;
        let mapped = unsafe {
            device.backend.device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()).ok()?
        };
        Some(Self {
            buf: GpuBuf { buffer: buf, memory: mem, size, mapped: mapped as *mut u8 },
            device: buf,
            size,
        })
    }
}

// ===========================================================================
// GpuComputeContext: high-level GPU compute orchestrator
// Wraps GpuDevice + ComputeGraph + pre-allocated buffers
// Each method adds ops to the graph; call submit_and_wait() to execute
// ===========================================================================

pub struct GpuComputeContext {
    pub device: Arc<GpuDevice>,
    pub weights: LayerWeights,
    pub kv_cache: Option<GpuKVCache>,
    pub scratch: Option<ScratchBuffers>,
    pub staging: Option<StagingBuffer>,
    pub node_count: usize,
    pub embed_dim: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub window_size: usize,
    pub enabled: bool,
}

unsafe impl Send for GpuComputeContext {}
unsafe impl Sync for GpuComputeContext {}

impl GpuComputeContext {
    pub fn new(
        device: &Arc<GpuDevice>,
        embed_dim: usize,
        ffn_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        window_size: usize,
        num_layers: usize,
    ) -> Option<Self> {
        if !device.enabled { return None; }
        let scratch = ScratchBuffers::new(device, embed_dim, ffn_dim, n_heads, window_size, head_dim)?;
        let staging = StagingBuffer::new(device, (embed_dim.max(ffn_dim) * 8) as u64)?;
        let kv_cache = GpuKVCache::new(device, n_kv_heads, window_size, head_dim)?;
        Some(Self {
            device: device.clone(),
            weights: LayerWeights::new(num_layers),
            kv_cache: Some(kv_cache),
            scratch: Some(scratch),
            staging: Some(staging),
            node_count: 0,
            embed_dim,
            ffn_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            window_size,
            enabled: true,
        })
    }

    pub fn is_operational(&self) -> bool {
        self.enabled
    }

    pub fn upload_layer_weights(
        &mut self,
        layer: usize,
        h_q: &[u8], h_k: &[u8], h_v: &[u8],
        h_o: &[u8], h_gate: &[u8], h_up: &[u8], h_down: &[u8],
        h_attn_norm: &[u8], h_ffn_norm: &[u8],
    ) -> bool {
        if !self.enabled { return false; }
        let upload = |h: &[u8]| -> Option<GpuBuf> {
            let backend = &self.device.backend;
            let size = h.len() as u64;
            let (buf, mem) = backend.allocate_buffer(
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ).ok()?;
            let staging = StagingBuffer::new(&self.device, size)?;
            staging.buf.copy_from_host(h);
            let cmd_alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(backend.compute_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cbs = unsafe { backend.device.allocate_command_buffers(&cmd_alloc).ok()? };
            let cb = cbs[0];
            unsafe {
                backend.device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok()?;
                let region = vk::BufferCopy::default().size(size).src_offset(0).dst_offset(0);
                backend.device.cmd_copy_buffer(cb, staging.buf.buffer, buf, &[region]);
                backend.device.end_command_buffer(cb).ok()?;
            }
            let fence = unsafe { backend.device.create_fence(&vk::FenceCreateInfo::default(), None).ok()? };
            let cbs_arr = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs_arr);
            let submits_arr = [submit];
            unsafe { backend.device.queue_submit(backend._queue, &submits_arr, fence).ok()?; }
            unsafe { backend.device.wait_for_fences(&[fence], true, u64::MAX).ok()?; }
            unsafe { backend.device.destroy_fence(fence, None); }
            Some(GpuBuf { buffer: buf, memory: mem, size, mapped: std::ptr::null_mut() })
        };
        let w = &mut self.weights;
        w.q[layer] = match upload(h_q) { Some(x) => x, None => return false };
        w.k[layer] = match upload(h_k) { Some(x) => x, None => return false };
        w.v[layer] = match upload(h_v) { Some(x) => x, None => return false };
        w.o[layer] = match upload(h_o) { Some(x) => x, None => return false };
        w.gate[layer] = match upload(h_gate) { Some(x) => x, None => return false };
        w.up[layer] = match upload(h_up) { Some(x) => x, None => return false };
        w.down[layer] = match upload(h_down) { Some(x) => x, None => return false };
        w.attn_norm[layer] = match upload(h_attn_norm) { Some(x) => x, None => return false };
        w.ffn_norm[layer] = match upload(h_ffn_norm) { Some(x) => x, None => return false };
        true
    }

    pub fn reset_graph(&mut self) {
        self.node_count = 0;
    }

    pub fn submit_and_wait(&mut self) -> bool {
        if !self.enabled || self.node_count == 0 { return false; }
        let mut cg = self.device.compute_graph.lock().unwrap();
        let pool = self.device.backend.compute_pool;
        let _ = cg.build(pool);
        let _ = cg.submit();
        let _ = cg.wait();
        cg.reset();
        self.node_count = 0;
        true
    }

    /// Add a GEMV Q4_K node to the compute graph. Returns node index.
    pub fn add_gemv_q4k_node(
        &mut self, w: vk::Buffer, input: vk::Buffer, output: vk::Buffer,
        n_rows: u32, n_blocks: u32, deps: Vec<usize>,
    ) -> usize {
        let mut cg = self.device.compute_graph.lock().unwrap();
        let idx = cg.add_node(crate::compute_graph::ComputeNodeOp::GEMVQ4K {
            d_w: w, d_x: input, d_out: output,
            n_rows, n_blocks,
        }, deps);
        self.node_count += 1;
        idx
    }

    /// Add an Attention node to the compute graph. Returns node index.
    pub fn add_attention_node(
        &mut self, d_q: vk::Buffer, d_k: vk::Buffer, d_v: vk::Buffer,
        d_scores: vk::Buffer, d_out: vk::Buffer,
        n_heads: u32, n_kv_heads: u32, seq_len: u32, head_dim: u32, kv_stride: u32,
        deps: Vec<usize>,
    ) -> usize {
        let mut cg = self.device.compute_graph.lock().unwrap();
        let idx = cg.add_node(crate::compute_graph::ComputeNodeOp::Attention {
            d_q, d_k, d_v, d_scores, d_out,
            n_heads, n_kv_heads, seq_len, head_dim, kv_stride,
        }, deps);
        self.node_count += 1;
        idx
    }

    /// Add a memory barrier node. Returns node index.
    pub fn add_barrier(&mut self, deps: Vec<usize>) -> usize {
        let mut cg = self.device.compute_graph.lock().unwrap();
        let idx = cg.add_node(crate::compute_graph::ComputeNodeOp::MemoryBarrier, deps);
        self.node_count += 1;
        idx
    }

    /// One-shot copy between buffers via staging (host-visible coherent)
    pub fn copy_between(&self, src: vk::Buffer, dst: vk::Buffer, src_off: u64, dst_off: u64, size: u64) -> bool {
        if !self.enabled { return false; }
        let backend = &self.device.backend;
        let pool = backend.compute_pool;
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cbs = match unsafe { backend.device.allocate_command_buffers(&cmd_alloc) } {
            Ok(c) => c, Err(_) => return false,
        };
        let cb = cbs[0];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if unsafe { backend.device.begin_command_buffer(cb, &begin) }.is_err() { return false; }
        let region = vk::BufferCopy::default()
            .src_offset(src_off).dst_offset(dst_off).size(size);
        unsafe { backend.device.cmd_copy_buffer(cb, src, dst, &[region]); }
        if unsafe { backend.device.end_command_buffer(cb) }.is_err() { return false; }
        let cbs_arr = [cb];
        let submit = vk::SubmitInfo::default().command_buffers(&cbs_arr);
        let submits_arr = [submit];
        let fence = match unsafe { backend.device.create_fence(&vk::FenceCreateInfo::default(), None) } {
            Ok(f) => f, Err(_) => return false,
        };
        let result = unsafe { backend.device.queue_submit(backend._queue, &submits_arr, fence) };
        if result.is_err() { return false; }
        let _ = unsafe { backend.device.wait_for_fences(&[fence], true, u64::MAX) };
        unsafe { backend.device.destroy_fence(fence, None); }
        true
    }

    pub fn upload_norm_weight(&mut self, layer: usize, is_attn: bool, h_bytes: &[u8]) -> bool {
        if !self.enabled { return false; }
        let backend = &self.device.backend;
        let size = h_bytes.len() as u64;
        let (buf, mem) = match backend.allocate_buffer(
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ) {
            Ok(x) => x, Err(_) => return false,
        };
        let staging = match StagingBuffer::new(&self.device, size) { Some(s) => s, None => return false };
        staging.buf.copy_from_host(h_bytes);
        let gpu = GpuBuf { buffer: buf, memory: mem, size, mapped: std::ptr::null_mut() };
        self.copy_between(staging.buf.buffer, buf, 0, 0, size);
        if is_attn {
            if layer < self.weights.attn_norm.len() { self.weights.attn_norm[layer] = gpu; }
        } else {
            if layer < self.weights.ffn_norm.len() { self.weights.ffn_norm[layer] = gpu; }
        }
        true
    }
}

// ===========================================================================
// Legacy CUDA Compatibility: stubs that always fall back to CPU
// Kept so executor.rs and scheduler.rs compile without changes
// ===========================================================================

#[derive(Debug, Clone, Copy)]
pub struct CudaStream(pub *mut std::ffi::c_void);
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

#[derive(Debug, Clone, Copy)]
pub struct CudaEvent(pub *mut std::ffi::c_void);
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

#[derive(Debug, Clone, Copy)]
pub struct AttentionGraph(pub *mut std::ffi::c_void);
unsafe impl Send for AttentionGraph {}
unsafe impl Sync for AttentionGraph {}

#[derive(Debug, Clone, Copy)]
pub struct LayerGraph(pub *mut std::ffi::c_void);
unsafe impl Send for LayerGraph {}
unsafe impl Sync for LayerGraph {}

#[derive(Debug, Clone, Copy)]
pub struct GemvGraph(pub *mut std::ffi::c_void);
unsafe impl Send for GemvGraph {}
unsafe impl Sync for GemvGraph {}

use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum GpuError {
    #[error("GPU not available (Vulkan fallback)")]
    NotAvailable(String),
}

pub type GpuResult<T> = std::result::Result<T, GpuError>;

pub fn gpu_init() -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_sync() -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_stream_create() -> GpuResult<CudaStream> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_stream_destroy(_: CudaStream) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_stream_synchronize(_: CudaStream) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_event_create() -> GpuResult<CudaEvent> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_event_destroy(_: CudaEvent) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_event_record(_: CudaEvent, _: *mut std::ffi::c_void) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_event_synchronize(_: CudaEvent) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_event_elapsed(_: CudaEvent, _: CudaEvent) -> GpuResult<f32> { Err(GpuError::NotAvailable("Vulkan backend".into())) }

pub unsafe fn gpu_alloc(_: usize) -> GpuResult<*mut std::ffi::c_void> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_free(_: *mut std::ffi::c_void) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_alloc_kv_buffer_half(_: usize, _: usize, _: usize) -> GpuResult<*mut std::ffi::c_void> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_free_weights(_: *mut u8) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }

pub fn gpu_upload_weights(_: *const u8, _: *mut *mut u8, _: usize, _: CudaStream) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_upload_weights_transposed(_: *const u8, _: *mut *mut u8, _: usize, _: i32, _: i32, _: CudaStream) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_gemv_q4k(_: *const u8, _: *const f32, _: *mut f32, _: i32, _: i32, _: CudaStream) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_gemv_q4k_full(_: *const u8, _: &[f32], _: &mut [f32], _: i32, _: i32, _: CudaStream) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_gemv_q4k_prealloc(_: *const u8, _: &[f32], _: &mut [f32], _: *mut f32, _: *mut f32, _: i32, _: i32, _: i32, _: i32, _: CudaStream) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_gemv_q6k(_: *const u8, _: *const f32, _: *mut f32, _: i32, _: i32, _: CudaStream) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }

pub fn gpu_alloc_buffers(_: *mut *mut f32, _: *mut *mut f32, _: i32, _: i32, _: CudaStream) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_free_buffers(_: *mut f32, _: *mut f32) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_copy_to_device_async(_: *mut std::ffi::c_void, _: *const std::ffi::c_void, _: usize, _: CudaStream) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_copy_to_host_async(_: *mut std::ffi::c_void, _: *const std::ffi::c_void, _: usize, _: CudaStream) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_copy_kv_layer_async_half(_: *mut std::ffi::c_void, _: &[f32], _: usize, _: usize, _: usize, _: usize, _: CudaStream) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_attention_streamed_half(_: *const f32, _: *const std::ffi::c_void, _: *const std::ffi::c_void, _: *mut f32, _: usize, _: usize, _: usize, _: usize, _: usize, _: CudaStream) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_graph_create_attention_half(_: *const f32, _: *const std::ffi::c_void, _: *const std::ffi::c_void, _: *mut f32, _: *mut f32, _: usize, _: usize, _: usize, _: usize, _: usize) -> GpuResult<AttentionGraph> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_graph_replay_attention(_: &AttentionGraph, _: CudaStream) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_graph_destroy(_: AttentionGraph) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_graph_create_gemv_q4k(_: *const u8, _: *const f32, _: *mut f32, _: i32, _: i32) -> GpuResult<GemvGraph> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_graph_create_gemv_q4k_qkv(_: *const u8, _: *const u8, _: *const u8, _: *const f32, _: *mut f32, _: *mut f32, _: *mut f32, _: i32, _: i32, _: i32, _: i32) -> GpuResult<GemvGraph> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_graph_create_gemv_q4k_gate_up(_: *const u8, _: *const u8, _: *const f32, _: *mut f32, _: *mut f32, _: i32, _: i32, _: i32) -> GpuResult<GemvGraph> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_graph_replay_gemv(_: &GemvGraph, _: CudaStream) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_graph_destroy_gemv(_: GemvGraph) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_attention_forward(_: &[f32], _: &[f32], _: &[f32], _: &mut [f32], _: usize, _: usize, _: usize, _: usize) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_event_elapsed_ms(_: CudaEvent, _: CudaEvent) -> GpuResult<f32> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_graph_create_layer(
    _: *mut u8, _: *mut u8, _: *mut u8, _: *mut u8, _: *mut u8, _: *mut u8, _: *mut u8,
    _: *mut f32, _: *mut f32,
    _: *mut std::ffi::c_void, _: *mut std::ffi::c_void,
    _: *mut f32, _: *mut f32, _: *mut f32, _: *mut f32, _: *mut f32,
    _: *mut f32, _: *mut f32, _: *mut f32, _: *mut f32, _: *mut f32, _: *mut f32,
    _: *mut i32, _: *mut i32,
    _: usize, _: usize, _: usize, _: usize, _: usize, _: usize,
    _: usize, _: usize, _: usize, _: usize, _: usize, _: usize, _: usize, _: usize,
    _: f32,
) -> GpuResult<LayerGraph> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_graph_replay_layer(_: &LayerGraph, _: CudaStream, _: usize, _: usize) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
pub fn gpu_graph_destroy_layer(_: LayerGraph) -> GpuResult<()> { Err(GpuError::NotAvailable("Vulkan backend".into())) }
