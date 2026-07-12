# Swamp — LLM Inference Engine

> Inferência de LLM em Rust + Vulkan, targeting GPU NVIDIA GTX 1650 4GB + CPU AVX-512 Tiger Lake.  
> Arquitetura **Control/Data Plane**: LuaJIT orquestra, Rust + Vulkan executam.

```
┌──────────────────────────────────────────────────────────────────┐
│                   CONTROL PLANE (LuaJIT)                          │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌───────────┐  │
│  │ Policy   │  │ Thermal      │  │ Governor   │  │ Pipeline  │  │
│  │ Engine   │──│ Coordinator  │──│ (AIMD+Arb) │──│ 7-stage   │  │
│  │ hot-reld │  │ RAPL+temp   │  │ auto-tune  │  │ concurrent│  │
│  └────┬─────┘  └──────────────┘  └────────────┘  └───────────┘  │
│       │ comandos via CommandQueue (ring buffer lock-free)         │
└───────┼──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│                    DATA PLANE (Rust + Vulkan)                     │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ HMA        │  │ ComputeGraph │  │ Vulkan Queue             │  │
│  │ (3 arenas) │──│ DAG builder  │──│ (timeline semaphore)     │  │
│  └────────────┘  └──────────────┘  └──────────────────────────┘  │
│       │                  │                                       │
│       ▼                  ▼                                       │
│  ┌────────────┐  ┌──────────────────┐                           │
│  │ Mojo CPU   │  │ GLSL/SPIR-V GPU  │                           │
│  │ (AVX-512)  │  │ (Vulkan Compute) │                           │
│  └────────────┘  └──────────────────┘                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## Stack

| Crate | Função | Tecnologia |
|-------|--------|------------|
| **swamp-engine** | Core inference: executor, scheduler, policy, thermal, SwampVM | Rust + LuaJIT |
| **swamp-gpu** | GPU backend: Vulkan, ComputeGraph DAG, Mojo loader | Vulkan + Mojo + GLSL |
| **swamp-kernels** | CPU SIMD kernels: Q4_K/Q6_K GEMV, Stokes KV cache, softmax | Rust (VNNI/AVX2/scalar) |
| **swamp-gguf** | GGUF parser + lazy dequantization, zero-copy mmap | Rust |
| **swamp-tensors** | Tensor types + HMA (3-tier memory allocator) | Rust |
| **swamp-server** | HTTP inference server (Axum + Prometheus) | Rust + LuaJIT |
| **swamp-tools** | CLI: benchmark, inspect | Rust |

---

## Architecture

### Control/Data Plane Separation

LuaJIT gerencia orquestração (decisões térmicas, políticas, hot-reload) **sem nunca tocar dados GPU**. Ele manipula handles opacos (`ResourceHandle(u64)`) e envia comandos via `CommandQueue` lock-free. O `DataPlane` thread processa a fila e constrói `ComputeGraph` DAGs para execução Vulkan.

### HMA — Heterogeneous Memory Allocator

Três arenas de memória gerenciadas pelo `HeterogeneousMemoryAllocator`:

| Arena | Localização | Uso |
|-------|-------------|-----|
| **HostPinned** | RAM (alinhado 64B) | SIMD CPU, Mojo kernels |
| **DeviceLocal** | VRAM (device-local) | Pesos e scratch GPU |
| **UnifiedMapped** | Memória compartilhada | iGPU/Apple Silicon zero-copy |

### ComputeGraph DAG

Em vez de dispatches ad-hoc por operação, o `ComputeGraph` constrói um **Directed Acyclic Graph** de dependências com memory barriers automáticas. Submissão via timeline semaphore:

```
┌─ GEMV Q4K ──┐     ┌─ Attention ──┐     ┌─ SiLU+Mul ──┐
│ d_w, d_x     │────→│ d_q, k, v    │────→│ gate, up    │────→│
│ d_out        │     │ d_out        │     │ out         │     │
└──────────────┘     └──────────────┘     └─────────────┘     │
                                                               │
┌──────────────────────────────────────────────────────────────┘
│ sync (1x total por submit)
▼
```

### Fused Layer Graph (12 ops → 1 submit)

```
① RMSNorm → ② GEMV Q → ③ GEMV K → ④ GEMV V → ⑤ RoPE →
⑥ KV save → ⑦ Attention → ⑧ GEMV O → ⑨ Add → ⑩ RMSNorm →
⑪ GEMV Gate+Up → ⑫ SiLU+Mul → ⑬ GEMV Down → ⑭ Add
```

**Antes:** 5 syncs/layer × 22 layers = **110 syncs/passo**  
**Depois:** 1 sync total (via ComputeGraph + timeline semaphore)  
**Fallback:** CPU por operação se GPU indisponível

---

## Performance

### Decode (TinyLlama 1.1B Q4_K_M, GTX 1650, 128 ctx)

| Configuração | ms/tok | tok/s | vs baseline |
|-------------|--------|-------|-------------|
| CPU-only (AVX-512) | 75.4 | 13 | 1× |
| + Ring dispatch | 38.0 | 26 | 2.0× |
| + GPU GEMV dispatch | ~19 | ~50 | ~4× |
| + ComputeGraph fused | ~15 | ~65 | ~5× |
| + Pipeline concurrente (--pipeline 4) | ~4 | ~250 | ~19× |

### KV Cache 1M Contexto

| Componente | Consumo |
|-----------|---------|
| KV cache f32 full attention | 44 GB ❌ |
| KV cache 4-bit + window=4096 VRAM | 22 MB/layer × 22 = 484 MB |
| Host PagedKVCache (1M ctx) | ~7 GB RAM |
| Modelo (Q4_K) | 549 MB |
| **Total VRAM** | **~1.1 GB** ✅ |

### Roofline — Gargalos por Operação

| Operação | MACs/layer | % total | Gargalo |
|----------|-----------|---------|---------|
| GEMV Gate | 11.53M | 26.2% | **Memory-bound** (Q4_K ~90 GB/s) |
| GEMV Up | 11.53M | 26.2% | **Memory-bound** |
| GEMV Down | 11.53M | 26.2% | **Memory-bound** |
| GEMV Q | 4.19M | 9.5% | Memory-bound |
| GEMV O | 4.19M | 9.5% | Memory-bound |
| GEMV K/V | 0.52M | 2.4% | Compute-bound |

---

## Hardware Target

| Parâmetro | GTX 1650 Mobile |
|-----------|-----------------|
| VRAM | 4 GB GDDR6 |
| SM count | 14 (Turing) |
| Compute capability | 7.5 |
| PCIe | 3.0 ×16 (~16 GB/s) |
| Shared mem/block | 48 KB |
| CPU | i5-11260H (Tiger Lake) |
| CPU PL1 | ~55W (térmico compartilhado) |

---

## Build

```bash
# Release (com GPU)
cargo build --release --features gpu

# Verificar erros
cargo build --release 2>&1 | grep "^error"

# Apenas GPU backend
cargo build --release -p swamp-gpu
```

### Shaders GLSL → SPIR-V

Se `glslc` (Vulkan SDK) disponível, os shaders em `swamp-gpu/shaders/` são compilados automaticamente:
- `gemv_q4k.comp` — GEMV com dequantização Q4_K
- `attention.comp` — FlashAttention online softmax
- `rmsnorm.comp` — RMS Normalization
- `rope.comp` — Rotary Position Embedding
- `silu_mul.comp` — SiLU + gating
- `add.comp` — Residual add

### Mojo Kernel

```bash
# Se mojo compiler disponível, o build.rs compila automaticamente
# swamp-gpu/kernels/attention.mojo → libswamp_mojo.so
# Fallback: CPU attention em Rust
```

### MSR Unlock (AVX-512 downclock)

```bash
sudo modprobe msr && sudo wrmsr -a 0x1FC 0x4004005f
```

---

## Benchmark

```bash
cargo run --release --features gpu -p swamp-tools --bin swamp-benchmark-prefill -- \
  /path/model.Q4_K_M.gguf /path/tokenizer.json \
  --prompt-tokens 128 --repeats 3 --n-threads 6
```

### Opções

| Flag | Default | Descrição |
|------|---------|-----------|
| `--prompt-tokens` | 128 | Tokens de prefill |
| `--repeats` | 1 | Repetições |
| `--n-threads` | 6 | Threads CPU |
| `--qat` | off | Calibração QAT |
| `--pipeline` | 1 | Pipeline parallelism |

---

## Dimensionamento (Modelos)

| Parâmetro | TinyLlama 1.1B |
|-----------|----------------|
| Embedding dim (d) | 2048 |
| Attention heads (h) | 32 |
| KV heads (h_kv) | 4 |
| Head dim (d_h) | 64 |
| FFN dim (d_ff) | 5632 |
| Camadas (L) | 22 |
| Vocab size (V) | 32000 |
| Pesos/layer | 44.0M MACs |
| Janela VRAM (W) | 4096 |

### Pesos por Camada (Q4_K)

| Projeção | Rows | Cols | Pesos | Bytes |
|----------|------|------|-------|-------|
| Q | 2048 | 2048 | 4.19M | 2.36 MB |
| K | 256 | 2048 | 0.52M | 0.29 MB |
| V | 256 | 2048 | 0.52M | 0.29 MB |
| O | 2048 | 2048 | 4.19M | 2.36 MB |
| Gate | 5632 | 2048 | 11.53M | 6.49 MB |
| Up | 5632 | 2048 | 11.53M | 6.49 MB |
| Down | 2048 | 5632 | 11.53M | 6.49 MB |

---

## Componentes do Engine

| Módulo | Arquivo | Função |
|--------|---------|--------|
| **Executor** | `executor.rs` | Decode loop, generate, prefill, dispatch GPU/CPU |
| **Scheduler** | `scheduler.rs` | PerLayerGpuState → GpuComputeContext bridge |
| **ControlPlane** | `control_plane.rs` | CommandQueue, ResourceHandle, OpaqueHandleStore |
| **DataPlane** | `data_plane.rs` | Thread consumidora + dispatch GpuComputeContext |
| **Policy** | `policy.rs` | LuaJIT Policy Engine hot-reloadable + gpu.* commands |
| **Linear** | `linear.rs` | GEMV linear layers: Q4_K/Q6_K dispatch, RingView |
| **Ops** | `ops.rs` | CPU: RMSNorm, RoPE (LUT), SiLU, attention, softmax |
| **Cache** | `cache.rs` | PagedKV Cache 3-tier (Hot FP32+Q4 / Cold NVMe) |
| **Fugu** | `fugu.rs` | Strategy orchestrator (attention, GEMV, cache, speculation) |
| **DSPark** | `dspark.rs` | LSH speculative decoding |
| **SwampVM** | `swamp_vm.rs` | Persistent opcode dispatcher (CPU/GPU/Cognitive) |
| **Governor** | `governor.rs` | ResourceGovernor: AIMD + PowerArbiter + Registry + Swapper |
| **Thermal** | `thermal.rs` | RAPL telemetry + throttle prediction |
| **Pipeline** | `pipeline.rs` | 7-stage multi-model pipeline |
| **QAT** | `qat.rs` | Quantization-Aware Training calibration |
| **AIMD** | `aimd.rs` | Additive Increase/Multiplicative Decrease |
| **VNpu** | `vnpu.rs` | Virtual NPU scheduler (EWMA budget tracking) |
| **HLC** | `hlc.rs` | Hybrid Logical Clock profiler |

---

### GPU Backend (swamp-gpu)

| Componente | Arquivo | Descrição |
|------------|---------|-----------|
| VkBackend | `vulkan.rs` | Instance, Device, Queue, Pipeline Cache |
| ComputeGraph | `compute_graph.rs` | DAG builder + timeline semaphore submit |
| ShaderCache | `shaders.rs` | SPIR-V module + pipeline cache |
| GpuDevice | `lib.rs` | DeviceAllocator impl + unified/device-local alloc |
| GpuComputeContext | `lib.rs` | High-level orchestrator: scratch, staging, weights |
| MojoKernel | `mojo.rs` | libswamp_mojo.so runtime loader |
| GpuBuf | `lib.rs` | vk::Buffer + vk::DeviceMemory + mapped ptr |
| HMA bridge | `hma.rs` | HostPinned/DeviceLocal/UnifiedMapped arenas |

---

### Governança Adaptativa

```
                    ┌──────────────────────────────────┐
                    │        ResourceGovernor           │
                    │  (unifica 5 subsistemas)          │
                    └──────────────────────────────────┘
           ┌───────────┬───────────┬───────────┬───────────┐
           ▼           ▼           ▼           ▼           ▼
     ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
     │AIMD     │ │Power    │ │Model    │ │Model    │ │Thermal  │
     │Ramp     │ │Arbiter  │ │Registry │ │Swapper  │ │LSC      │
     ├─────────┤ ├─────────┤ ├─────────┤ ├─────────┤ ├─────────┤
     │+500ms   │ │RAPL+temp│ │VRAM/RAM │ │LRU swap │ │Freq     │
     │dobra    │ │split    │ │CPU-only │ │prefetch │ │RAPL     │
     │corta na │ │CPU/iGPU │ │tiers    │ │pipeline │ │throttle │
     │1ª falha │ │         │ │         │ │         │ │detect   │
     └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
           │           │           │           │           │
           └───────────┴───────────┴───────────┴───────────┘
                                ▼
                     ┌──────────────────────┐
                     │  SwampVM dispatcher  │
                     │  CPU / GPU / Cog     │
                     └──────────────────────┘
```

---

## Licença

MIT
