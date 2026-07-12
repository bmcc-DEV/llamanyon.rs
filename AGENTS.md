# Build & Benchmark

## Build
```bash
cargo build --release -p swamp-tools --bin swamp-benchmark-prefill
```

## Build with GPU
```bash
cd swamp-gpu/kernels && nvcc -O2 --std=c++17 -arch=sm_75 -shared -Xcompiler "-fPIC" \
  fused_attention.cu -o ../libswamp_gpu.so && cd ../..
cargo build --release --features gpu -p swamp-engine -p swamp-tools
```

## Run benchmark (decode + prefill)
```bash
cargo run --release -p swamp-tools --bin swamp-benchmark-prefill -- \
  /media/bruno/3e94d163-2a59-473e-bcc5-09148350a987/MODELS/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  /media/bruno/3e94d163-2a59-473e-bcc5-09148350a987/MODELS/tokenizer.json \
  --prompt-tokens 128 --repeats 3 --n-threads 6
```

Options:
- `--n-threads N` — thread count (default: 6, try 4-6)
- `--qat` — enable QAT calibration before decode
- `--repeats N` — benchmark repeat count

## Power tuning (may help reduce AVX-512 downclock)
```bash
sudo modprobe msr && sudo wrmsr -a 0x1FC 0x4004005f
```

## Build all crates (check for errors)
```bash
cargo build --release 2>&1 | grep "^error"
```

## Detailed profiler output
The benchmark prints per-layer timing breakdown:
- GEMV QKV: Q/K/V projection GEMV group
- GEMV O: output projection GEMV
- GEMV GateUp: FFN gate + up GEMVs (dominant bottleneck ~43%)
- GEMV Down: FFN down projection GEMV
- Attention: scaled dot-product attention
- RMSNorm: RMS normalization

## Fused Layer Graph (GPU only)
One CUDA Graph captures the entire transformer layer: RMSNorm → QKV → RoPE → KV save → Attention → O → add → RMSNorm → GateUp → SiLU+Mul → Down → add.
- 12 CUDA kernels fused into 1 graph
- seq_len and wpos passed via device pointers (updated before each replay)
- Eliminates 4+ H2D/D2H copies and 5 syncs per layer
- Fallback: individual per-op GPU graphs → CPU

## Key files
- `swamp-gpu/kernels/fused_attention.cu` — all GPU kernels + layer graph API (C extern)
- `swamp-gpu/src/lib.rs` — Rust FFI for `LayerGraph` and all GPU functions
- `swamp-engine/src/scheduler.rs` — `PerLayerGpuState` with fused graph management
- `swamp-engine/src/executor.rs` — decode loop with fused graph path
- `swamp-kernels/src/fused_gemv_q4k.rs` — VNNI/AVX2/scalar GEMV kernels
- `swamp-tools/src/bin/benchmark_prefill.rs` — benchmark harness
