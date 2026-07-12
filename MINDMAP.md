# Swamp — Mapa Mental

> Inferência de LLM em Rust + Vulkan para GTX 1650 4GB + CPU AVX-512.
> Branch: `experimental-1m` — github.com/bmcc-DEV/swamp

---

## 1. ⚠️ O PROBLEMA CENTRAL: onde o Forth VM se encaixa no loop

**Diagrama proposto: Lua → Forth → Rust → Hardware**

Se Forth roda **por token** (sensor-read-vram / adaptive-gemv / execute-layer a cada passo), é regressão de performance:

| Abordagem | Custo por decisão |
|-----------|------------------|
| `AimdRamp::adapt()` (Rust nativo) | chamada inlineável, zero alocação |
| Forth interpretado por token | push/pop stack, dispatch, sem inline |

→ Trocar decisões já resolvidas em Rust nativo por um interpretador stack machine **por token** é pagar overhead de interpretação disfarçado de arquitetura nova.

**Linha vermelha:** nenhum interpretador no hot path do decode loop.

---

## 2. ✅ "ORGANISMO ADAPTATIVO" JÁ EXISTE

Sensor → decisão adaptativa → feedback já roda em Rust, sem custo de interpretação:

```
AimdRamp::adapt()      ← additive-increase/multiplicative-decrease
ThermalLSC::poll()     ← frequência + RAPL + throttle prediction
PowerArbiter::decide() ← split CPU/iGPU por telemetria
```

O nome bonito ("organismo vivo") não justifica reimplementar em Forth o que já roda mais rápido em Rust.

---

## 3. 🎯 ONDE FORTH GENUINAMENTE GANHA

**Escopo estreito, não o loop principal.**

| Uso | Custo | Vale a pena? |
|-----|-------|-------------|
| Decidir GEMV path por token | 1 chamada Rust inlineável ≤ 5ns | ❌ |
| Hot-patch de política raro | 1× na carga do modelo / reload Lua | ✅ |
| Usuário registrar `custom_kernel.forth` | 1×, cacheado, vira flag enum Rust | ✅ |

**Regra:** Forth só deve rodar na **mesma cadência do PolicyEngine Lua** (carga de modelo, reload de policy). Resultado vira flag/enum consumido pelo Rust no hot path — sem interpretação por token.

---

## 4. 🏗️ STACK TECHNOLOGY

```
┌────────────────────────────────────────────────────────────┐
│  swamp-server  (axum HTTP + Prometheus + Lua policy)       │
├────────────────────────────────────────────────────────────┤
│  swamp-engine  (core inference engine)                     │
│    ├── Executor: generate/generate_batch, decode loop      │
│    ├── Scheduler: GPU streams, graphs, PerLayerGpuState    │
│    ├── Governor: AIMD + Power + Registry + Swapper + LSC   │
│    ├── SwampVM: ring MPSC + CPU/GPU/Cognitive backends     │
│    ├── Pipeline: 7-stage multi-model graph                 │
│    ├── DSPark: speculative decoding (LSH pattern draft)    │
│    └── +20 modules (ops, cache, sampler, policy, etc.)     │
├────────────────────────────────────────────────────────────┤
│  swamp-gpu  (CUDA FFI — libloading runtime)                │
│    ├── fused_attention.cu — 12+ kernels, sm_75             │
│    ├── LayerGraph — 12 kernels em 1 cudaGraphExec          │
│    ├── GemvGraph — QKV / GateUp / Single                   │
│    └── AttentionGraph — por seq_len cache                   │
├────────────────────────────────────────────────────────────┤
│  swamp-kernels  (CPU GEMV — VNNI / AVX2 / scalar)          │
├────────────────────────────────────────────────────────────┤
│  swamp-gguf  (GGUF parser — zero-copy mmap, dequant)       │
├────────────────────────────────────────────────────────────┤
│  swamp-tools  (benchmarks + test CLI)                      │
└────────────────────────────────────────────────────────────┘
```

---

## 5. 🧬 ARQUITETURA EM CAMADAS

```
                    ┌──────────────────────┐
                    │   HTTP API (axum)     │
                    │  Prometheus + Lua     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │    Policy Engine     │
                    │   (Lua hot-reload)   │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
  ┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
  │  SwampVM      │   │  Governor     │   │  Pipeline     │
  │ (opcode ring) │   │ (admissão)    │   │ (multi-model) │
  │ CPU/GPU/Cog   │   │ auto-tune     │   │ 7 stages      │
  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │   ModelExecutor      │
                    │   generate() / loop  │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
  ┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
  │  Scheduler    │   │  PerLayerGpu  │   │  Cache        │
  │  GpuStreams   │   │  State Pesos  │   │  TieredPaged  │
  │  Graphs       │   │  Buffers      │   │  KV (3 tiers) │
  └───────┬───────┘   └───────┬───────┘   └───────────────┘
          │                    │
          └────────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │  GPU / CPU kernels   │
                    │  fused_attention.cu  │
                    │  fused_gemv_q4k.rs   │
                    └──────────────────────┘
```

---

## 6. ⚡ FLUXO DE EXECUÇÃO (DECODE)

### All-Fused (1 sync total)

```
upload_x ─→ [Layer 0 fused graph] ─→ [Layer 1] ─→ ... ─→ [Layer N] ─→ sync + download_x
              └── 12 kernels CUDA em 1 cudaGraphExec ──┘
```

### 12 kernels capturados por layer graph

```
① rmsnorm_attn(x, d_attn_norm[l])       → x_norm
② GEMV Q(x_norm, d_w_q[l])              → q_out
③ GEMV K(x_norm, d_w_k[l])              → k_out
④ GEMV V(x_norm, d_w_v[l])              → v_out
⑤ RoPE(q_out, k_out, pos)               → in-place
⑥ KV save half(k_out, v_out, d_k_buf, wpos) → ring buffer
⑦ attention(q_out, d_k_buf, d_v_buf, seq_len) → attn_out
⑧ GEMV O(attn_out, d_w_o[l])            → o_out
⑨ add(x, o_out)                          → x += o_out
⑩ rmsnorm_ffn(x, d_ffn_norm[l])         → x_norm
⑪ GEMV Gate(x_norm, d_w_gate[l])        → gate_out
⑫ GEMV Up(x_norm, d_w_up[l])            → up_out
⑬ silu_mul(gate_out, up_out)             → gate_out *= silu(gate_out) * up_out
⑭ GEMV Down(gate_out, d_w_down[l])      → down_out
⑮ add(x, down_out)                       → x += down_out
```

### Replay: só precisa do `pos`

```
Antes: 110 syncs (5 syncs × 22 layers)
Depois: 1 sync (cudaMemcpyAsync d_wpos/d_seq_len → cudaGraphLaunch)
```

### Fallback (se fused falha)

```
per-op GPU (GEMV graph async) → CPU kernel (RoPE, RMSNorm, SiLU, Add)
```

---

## 7. 🤖 SISTEMA DE GOVERNANÇA ADAPTATIVA

```
                    ┌──────────────────────────────────┐
                    │        ResourceGovernor          │
                    │  (unifica 5 subsistemas)         │
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

**O que cada subsistema faz:**

| Subsistema | Sensor | Decisão | Ação |
|-----------|--------|---------|------|
| AIMD Ramp | Timestamp | Dobra/corta budget | `allow_inference()` retorna bool |
| PowerArbiter | RAPL + temp | Split CPU/iGPU | Ajusta thread count + clock |
| ModelRegistry | Carga de modelo | Qual tier? | VRAM / RAM / CPU-only |
| ModelSwapper | LRU age + prefetch | Swap in/out | Move pesos entre VRAM ↔ RAM |
| ThermalLSC | Frequência + RAPL | Throttle detect | Reduz intensidade |

---

## 8. 📦 CATÁLOGO DE COMPONENTES

### swamp-engine/src (core — 27 módulos)

| Módulo | Função |
|--------|--------|
| **executor.rs** | Laço de inferência: generate(), generate_batch(), dispatch GPU |
| **scheduler.rs** | GpuStreamManager, PerLayerGpuState (13 buffers scratch, graphs por layer) |
| **linear.rs** | forward_gemvs_ring, SENSITIVITY_MAP (precisão adaptativa por linha) |
| **ops.rs** | CPU kernels: rmsnorm, silu, add, mul, rope (LUT-based) |
| **cache.rs** | TieredPagedKVCache: Hot (Q4 RAM) / FP32 RAM / Cold NVMe |
| **model.rs** | Carregador de tensores GGUF + metadados |
| **tokenizer.rs** | Tokenizers wrapper (OnceLock global) |
| **sampler.rs** | Top-k, top-p, temperatura |

#### Governança

| Módulo | Função |
|--------|--------|
| **governor.rs** | ResourceGovernor + CognitiveWorkerPool |
| **aimd.rs** | AIMD: additive-increase / multiplicative-decrease |
| **power_arbiter.rs** | RAPL + temperatura → split CPU/iGPU |
| **thermal.rs** | ThermalLSC: frequência + RAPL + throttle prediction |
| **lsc.rs** | LscPrefetcher: curva de ganho baseada em contexto |
| **prefetch.rs** | PrefetchEngine (mmap_ptr + mmap_len) |

#### Swarm / VM

| Módulo | Função |
|--------|--------|
| **swamp_vm.rs** | SwampVM: ring MPSC + CPU/GPU/Cognitive backends + SessionContext |
| **vnpu.rs** | VirtualNpuScheduler: EWMA budget tracking |
| **pipeline.rs** | Multi-model pipeline: Qwen, DeepSeek, GLM, Phi, SmolVLM, Llama 4, GLM-4.5 |

#### Estratégia / Especulação

| Módulo | Função |
|--------|--------|
| **fugu.rs** | Strategy orchestrator (atenção, GEMV, cache, speculative) |
| **dspark.rs** | DSPark: speculative decoding com LSH pattern matching |
| **execution_moe.rs** | ExpertRouter singleton para GEMV |
| **virtual_experts.rs** | MoE Expert Prefetch (hash LSH do DSPark) |

#### Auxiliares

| Módulo | Função |
|--------|--------|
| **policy.rs** | LuaJIT Policy Engine hot-reloadable |
| **staging.rs** | StagingBuffer decoupled (workers → RAM sem sync) |
| **qat.rs** | Calibração 4-bit por bloco (activation stats) |
| **hlc.rs** | Hybrid Logical Clock + profiling init |
| **timewarp.rs** | Clock virtual 0.5x-3.0x |
| **chat_template.rs** | Formatação de prompt |

### swamp-gpu (CUDA FFI — 60+ exports)

| API | Função |
|-----|--------|
| `gpu_init/sync/available` | Lifecycle |
| `CudaStream` create/destroy/sync | Stream management |
| `CudaEvent` create/record/sync/elapsed | Event timing |
| `gpu_upload_weights/free_weights` | Weight management |
| `gpu_alloc_buffers/free_buffers` | Scratch buffers |
| `gpu_attention_forward/device/streamed` | Attention (f32 + half) |
| `gpu_gemv_q4k/full/prealloc` | GEMV (sync + async) |
| `gpu_swamp_init/launch/enqueue/readback/shutdown` | Swamp continuum |
| `LayerGraph` create/replay/destroy | Fused layer graph |
| `GemvGraph` create/replay/destroy | GEMV graphs (QKV, GateUp, Single) |
| `AttentionGraph` create/replay/destroy | Attention graph cache |

### swamp-kernels (CPU)

| Kernel | ISA |
|--------|-----|
| `fused_gemv_q4k` VNNI | AVX-512 (VPDPBUSD) |
| `fused_gemv_q4k` AVX2 | AVX2 (VPMADDUBSW) |
| `fused_gemv_q4k` scalar | Fallback |

---

## 9. 📊 PERFORMANCE (TinyLlama 1.1B Q4_K_M, GTX 1650, 128 ctx)

### Decode

| Configuração | ms/tok | tok/s | vs baseline |
|-------------|--------|-------|-------------|
| CPU-only (AVX-512) | 75.4 | 13 | 1× |
| + Ring dispatch | 38.0 | 26 | 2.0× |
| + GPU GEMV dispatch (7/layer) | ~19 | ~50 | ~4× |
| + CUDA Graph fused (4/layer) | ~15 | ~65 | ~5× |
| + Pipeline concurrent (--pipeline 4) | ~4 | ~250 | ~19× |

### KV Cache 1M contexto

| Componente | Consumo |
|-----------|---------|
| KV cache f32 full attention | 44 GB ❌ |
| KV cache 4-bit + window=4096 VRAM | 22 MB/layer × 22 = 484 MB |
| Host PagedKVCache (1M ctx) | ~7 GB RAM |
| Modelo (Q4_K) | 549 MB |
| **Total VRAM** | **~1.1 GB** ✅ (cabe na GTX 1650 4GB) |

### Sync reduction

| Antes | Depois |
|-------|--------|
| 5 syncs/layer × 22 layers = **110 syncs/passo** | **1 sync total** |

---

## 10. 💻 HARDWARE TARGET

| Parâmetro | GTX 1650 |
|-----------|----------|
| VRAM | 4 GB |
| SM count | 14 |
| Compute capability | 7.5 (Turing) |
| PCIe | 3.0 ×16 (~16 GB/s) |
| Shared mem/block | 48 KB |
| CPU PL1 | ~55W (térmico compartilhado) |

---

## 11. 🚀 COMO COMEÇAR

### Build GPU
```bash
cd swamp-gpu/kernels && nvcc -O2 --std=c++17 -arch=sm_75 -shared -Xcompiler "-fPIC" \
  fused_attention.cu -o ../libswamp_gpu.so && cd ../..
cargo build --release --features gpu -p swamp-engine -p swamp-tools
```

### Benchmark
```bash
cargo run --release --features gpu -p swamp-tools --bin swamp-benchmark-prefill -- \
  /path/model.Q4_K_M.gguf /path/tokenizer.json \
  --prompt-tokens 128 --repeats 3
```

### Pipeline concurrente
```bash
cargo run --release --features gpu -p swamp-tools --bin swamp-benchmark-prefill -- \
  /path/model.Q4_K_M.gguf /path/tokenizer.json \
  --prompt-tokens 128 --pipeline 4
```

### MSR unlock (AVX-512)
```bash
sudo modprobe msr && sudo wrmsr -a 0x1FC 0x4004005f
```

---

## 12. 📚 DOCUMENTAÇÃO

| Arquivo | Conteúdo |
|---------|----------|
| `README.md` | Stack diagram, performance, build/uso CLI |
| `TECHNICAL.md` | Sync analysis, fused graph FFI API, roadmap, hardware |
| `AGENTS.md` | Build/benchmark commands, power tuning, key files |
| `docs/RUNNER_1B_PLAN.md` | Roadmap completo para 1B runner |
| `docs/VNPU_CONCEPT.md` | VNPU schedulers conceitual |
