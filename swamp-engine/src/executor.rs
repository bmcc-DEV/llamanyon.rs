// swamp-engine/src/executor.rs
// Executor: laço de inferência de tokens que envia resultados via mpsc channel

use crate::model::Model;

use crate::sampler::Sampler;
use crate::thermal::ThermalCoordinator;
use crate::lsc::LscPrefetcher;
use crate::policy::PolicyEngine;
use crate::fugu::{FuguOrchestrator, SystemSnapshot, AttentionStrategy};
use crate::dspark::DSparkEngine;
use tokio::sync::mpsc::Sender;
use std::sync::Arc;
use crate::cache::PagedKVCache;

pub struct InferenceRequest {
    pub prompt: Option<String>,
    pub messages: Option<Vec<crate::chat_template::ChatMessage>>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

use rayon::ThreadPool;
use std::sync::OnceLock;

pub static RAYON_POOL: OnceLock<ThreadPool> = OnceLock::new();

pub fn get_rayon_pool() -> &'static ThreadPool {
    RAYON_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(6)
            .thread_name(|i| format!("swamp-vnni-{}", i))
            .build()
            .expect("failed to build Rayon thread pool")
    })
}

/// Batch prefill helper — process a contiguous slice of prompt tokens.
/// Returns the last token's hidden state.
#[allow(unused_variables)]
fn prefill_slice(
    num_layers: usize,
    embed_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    rms_eps: f32,
    tokens: &[usize],
    token_embd: &[f32],
    attn_norms: &[Vec<f32>],
    ffn_norms: &[Vec<f32>],
    model: &Model,
    kv_cache: &mut PagedKVCache,
    #[cfg(feature = "gpu")] per_layer_gpu: &mut Option<crate::scheduler::PerLayerGpuState>,
    pos_offset: usize,
    n_threads: usize,
) -> Vec<f32> {
    use crate::linear::{forward_linear_batch};
    use crate::ops::{rmsnorm, silu, add_in_place, mul_in_place, apply_rope_ufc};
    use rayon::prelude::*;

    let batch = tokens.len();
    let mut xs: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; embed_dim]).collect();
    let mut x_norms: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; embed_dim]).collect();
    let mut qs: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; num_heads * head_dim]).collect();
    let mut ks: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; num_kv_heads * head_dim]).collect();
    let mut vs: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; num_kv_heads * head_dim]).collect();
    let mut attn_outs: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; embed_dim]).collect();
    let mut wo_outs: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; embed_dim]).collect();
    let mut ffn_gates: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; ffn_dim]).collect();
    let mut ffn_ups: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; ffn_dim]).collect();
    let mut ffn_downs: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; embed_dim]).collect();

    xs.par_iter_mut().enumerate().for_each(|(i, x)| {
        let tok = tokens[i];
        let embd_start = tok * embed_dim;
        x.copy_from_slice(&token_embd[embd_start..embd_start + embed_dim]);
    });

    for l in 0..num_layers {
        let attn_w = &attn_norms[l];
        let ffn_w = &ffn_norms[l];

        // 1. Batched RMSNorm + QKV linear
        {
            x_norms.par_iter_mut().zip(xs.par_iter()).for_each(|(xn, x)| {
                rmsnorm(xn, x, attn_w, rms_eps);
            });

            let q_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_q.weight", l)).unwrap();
            let k_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_k.weight", l)).unwrap();
            let v_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_v.weight", l)).unwrap();

            let xn_refs: Vec<&[f32]> = x_norms.iter().map(|v| v.as_slice()).collect();
            let mut q_refs: Vec<&mut [f32]> = qs.iter_mut().map(|v| v.as_mut_slice()).collect();
            forward_linear_batch(&model.gguf, q_t, &xn_refs, &mut q_refs, n_threads).ok();
            let mut k_refs: Vec<&mut [f32]> = ks.iter_mut().map(|v| v.as_mut_slice()).collect();
            forward_linear_batch(&model.gguf, k_t, &xn_refs, &mut k_refs, n_threads).ok();
            let mut v_refs: Vec<&mut [f32]> = vs.iter_mut().map(|v| v.as_mut_slice()).collect();
            forward_linear_batch(&model.gguf, v_t, &xn_refs, &mut v_refs, n_threads).ok();
        }

        // 2. RoPE + KV cache store (position = pos_offset + t)
        for t in 0..batch {
            let abs_pos = pos_offset + t;
            apply_rope_ufc(&mut qs[t], &mut ks[t], abs_pos, num_heads, num_kv_heads, head_dim, model.config.context_len);
            kv_cache.save_at(l, abs_pos, &ks[t], &vs[t]);
        }

        // 3. Attention loop (seq_len = pos_offset + t + 1)
        let use_gpu = {
            #[cfg(feature = "gpu")]
            { per_layer_gpu.is_some() }
            #[cfg(not(feature = "gpu"))]
            { false }
        };
        if use_gpu {
            #[cfg(feature = "gpu")]
            if let Some(gpu) = per_layer_gpu {
                for t in 0..batch {
                    let abs_pos = pos_offset + t;
                    let seq_len = abs_pos + 1;
                    gpu.upload_kv_async(&ks[t], &vs[t], abs_pos);
                    gpu.execute_attention_async(&qs[t], &mut attn_outs[t], abs_pos, seq_len);
                    gpu.sync();
                }
            }
        } else {
            for t in 0..batch {
                let abs_pos = pos_offset + t;
                let seq_len = abs_pos + 1;
                kv_cache.ensure_pages_hot(0, kv_cache.page_id(seq_len.saturating_sub(1)));
                crate::ops::attention(&mut attn_outs[t], &qs[t], kv_cache, l, seq_len, abs_pos, num_heads, num_kv_heads, head_dim);
            }
        }

        // 4. Batched output projection + residual
        {
            let o_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_output.weight", l)).unwrap();
            let attn_refs: Vec<&[f32]> = attn_outs.iter().map(|v| v.as_slice()).collect();
            let mut wo_refs: Vec<&mut [f32]> = wo_outs.iter_mut().map(|v| v.as_mut_slice()).collect();
            forward_linear_batch(&model.gguf, o_t, &attn_refs, &mut wo_refs, n_threads).ok();
        }

        xs.par_iter_mut().zip(wo_outs.par_iter()).for_each(|(x, wo)| {
            add_in_place(x, wo);
        });

        // 5. Batched FFN
        {
            x_norms.par_iter_mut().zip(xs.par_iter()).for_each(|(xn, x)| {
                rmsnorm(xn, x, ffn_w, rms_eps);
            });

            let xn_refs: Vec<&[f32]> = x_norms.iter().map(|v| v.as_slice()).collect();
            let gate_t = model.gguf.tensor_or_err(&format!("blk.{}.ffn_gate.weight", l)).unwrap();
            let up_t = model.gguf.tensor_or_err(&format!("blk.{}.ffn_up.weight", l)).unwrap();
            let mut gate_refs: Vec<&mut [f32]> = ffn_gates.iter_mut().map(|v| v.as_mut_slice()).collect();
            forward_linear_batch(&model.gguf, gate_t, &xn_refs, &mut gate_refs, n_threads).ok();
            let mut up_refs: Vec<&mut [f32]> = ffn_ups.iter_mut().map(|v| v.as_mut_slice()).collect();
            forward_linear_batch(&model.gguf, up_t, &xn_refs, &mut up_refs, n_threads).ok();
        }

        for t in 0..batch {
            silu(&mut ffn_gates[t]);
            mul_in_place(&mut ffn_gates[t], &ffn_ups[t]);
        }

        let down_t = model.gguf.tensor_or_err(&format!("blk.{}.ffn_down.weight", l)).unwrap();
        let ffn_gate_refs: Vec<&[f32]> = ffn_gates.iter().map(|v| v.as_slice()).collect();
        let mut down_refs: Vec<&mut [f32]> = ffn_downs.iter_mut().map(|v| v.as_mut_slice()).collect();
        forward_linear_batch(&model.gguf, down_t, &ffn_gate_refs, &mut down_refs, n_threads).ok();

        xs.par_iter_mut().zip(ffn_downs.par_iter()).for_each(|(x, fd)| {
            add_in_place(x, fd);
        });
    }

    xs.into_iter().last().unwrap_or_default()
}

const PREFILL_CHUNK_SIZE: usize = 4096;

/// Batched prefill: process all prompt tokens, chunked to avoid OOM.
/// Each chunk runs the full layer pipeline (QKV, RoPE, save, attention, output, FFN)
/// with correct absolute positions, accumulating the KV cache across chunks.
/// Returns the last token's hidden state (x) for decode to continue from.
pub fn prefill_batch(
    num_layers: usize,
    embed_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    rms_eps: f32,
    prompt_tokens: &[usize],
    token_embd: &[f32],
    attn_norms: &[Vec<f32>],
    ffn_norms: &[Vec<f32>],
    model: &Model,
    kv_cache: &mut PagedKVCache,
    #[cfg(feature = "gpu")] per_layer_gpu: &mut Option<crate::scheduler::PerLayerGpuState>,
    n_threads: usize,
) -> Vec<f32> {
    let batch = prompt_tokens.len();
    if batch <= PREFILL_CHUNK_SIZE {
        return prefill_slice(
            num_layers, embed_dim, num_heads, num_kv_heads, head_dim, ffn_dim, rms_eps,
            prompt_tokens, token_embd, attn_norms, ffn_norms, model, kv_cache,
            #[cfg(feature = "gpu")] per_layer_gpu,
            0, n_threads,
        );
    }

    let mut last_x = vec![0.0f32; embed_dim];
    let mut offset = 0;
    while offset < batch {
        let end = (offset + PREFILL_CHUNK_SIZE).min(batch);
        let chunk = &prompt_tokens[offset..end];
        last_x = prefill_slice(
            num_layers, embed_dim, num_heads, num_kv_heads, head_dim, ffn_dim, rms_eps,
            chunk, token_embd, attn_norms, ffn_norms, model, kv_cache,
            #[cfg(feature = "gpu")] per_layer_gpu,
            offset, n_threads,
        );
        offset = end;
    }
    last_x
}

pub struct ModelExecutor {
    pub model: Arc<Model>,
    pub thermal_coordinator: ThermalCoordinator,
    pub prefetcher: LscPrefetcher,
    pub policy: PolicyEngine,
    // Tensor cache for non-linear operations (RMSNorm weights and embeddings)
    pub token_embd: Vec<f32>,
    pub attn_norms: Vec<Vec<f32>>,
    pub ffn_norms: Vec<Vec<f32>>,
    pub output_norm: Vec<f32>,
}

impl Clone for ModelExecutor {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            thermal_coordinator: ThermalCoordinator::default(),
            prefetcher: LscPrefetcher::default(),
            policy: PolicyEngine::new(&self.policy.script_path().to_string(), 6)
                .unwrap_or_else(|_| PolicyEngine::new("", 6).unwrap()),
            token_embd: self.token_embd.clone(),
            attn_norms: self.attn_norms.clone(),
            ffn_norms: self.ffn_norms.clone(),
            output_norm: self.output_norm.clone(),
        }
    }
}

impl ModelExecutor {
    pub fn new(model: Arc<Model>) -> Self {
        let head_dim = model.config.embed_dim / model.config.num_heads;
        crate::ops::init_rope_lut(head_dim, model.config.context_len);
        let thermal_coordinator = ThermalCoordinator::default();
        let prefetcher = LscPrefetcher::default();

        // Carga de pesos 1D para F32 (normas e embeddings)
        let token_embd = model.gguf.dequantize_tensor_alloc(model.gguf.tensor_or_err("token_embd.weight").unwrap()).unwrap();
        
        let num_layers = model.config.num_layers;
        let mut attn_norms = Vec::with_capacity(num_layers);
        let mut ffn_norms = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let attn_norm = model.gguf.dequantize_tensor_alloc(model.gguf.tensor_or_err(&format!("blk.{}.attn_norm.weight", l)).unwrap()).unwrap();
            let ffn_norm = model.gguf.dequantize_tensor_alloc(model.gguf.tensor_or_err(&format!("blk.{}.ffn_norm.weight", l)).unwrap()).unwrap();
            attn_norms.push(attn_norm);
            ffn_norms.push(ffn_norm);
        }
        let output_norm = model.gguf.dequantize_tensor_alloc(model.gguf.tensor_or_err("output_norm.weight").unwrap()).unwrap();

        // Load LuaJIT thermal policy; fall back to defaults if file not found
        let policy_paths = [
            "policies/default.lua",
            "../policies/default.lua",
            "/etc/swamp/policy.lua",
        ];
        let policy = policy_paths.iter().find_map(|path| {
            PolicyEngine::new(path, 6).ok()
        }).unwrap_or_else(|| PolicyEngine::new("", 6).unwrap()); // empty path = no script, all fallbacks

        Self {
            model,
            thermal_coordinator,
            prefetcher,
            policy,
            token_embd,
            attn_norms,
            ffn_norms,
            output_norm,
        }
    }

    pub async fn generate(&mut self, req: InferenceRequest, tx: Sender<String>) -> anyhow::Result<()> {
        use std::time::Instant;

        let sampler = Sampler::new(req.temperature, req.top_k, req.top_p);
        let prompt_str = if let Some(msgs) = &req.messages {
            crate::chat_template::format_chat_prompt(msgs)
        } else if let Some(p) = &req.prompt {
            p.clone()
        } else {
            String::new()
        };

        let prompt_tokens = if prompt_str.is_empty() {
            vec![1usize]
        } else {
            crate::tokenizer::encode(&prompt_str)
        };

        println!("DEBUG: Formatted prompt:\n{}\nTokens: {:?}", prompt_str, prompt_tokens);

        let num_layers = self.model.config.num_layers;
        let embed_dim = self.model.config.embed_dim;
        let num_heads = self.model.config.num_heads;
        let num_kv_heads = self.model.config.num_kv_heads;
        let head_dim = embed_dim / num_heads;
        let rms_eps = 1e-5;

        // Determine FFN dim from the first layer's gate weight
        let ffn_gate_t = self.model.gguf.tensor_or_err("blk.0.ffn_gate.weight")?;
        let ffn_dim = ffn_gate_t.shape[1] as usize;

        // Clone references to read-only buffers so we can move them into spawn_blocking
        let token_embd = self.token_embd.clone();
        let attn_norms = self.attn_norms.clone();
        let ffn_norms = self.ffn_norms.clone();
        let output_norm = self.output_norm.clone();
        let model = self.model.clone();

        let policy_path = self.policy.script_path().to_string();

        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            use crate::linear::{forward_linear, forward_linear_multi};
            use crate::ops::{rmsnorm, silu, add_in_place, mul_in_place, apply_rope_ufc};

            // Thermal telemetry: read fresh inside blocking section (Idea #2)
            let max_freq_khz = 4_400_000; // Tiger Lake-H cpuinfo_max_freq
            let mut temp_celsius = crate::thermal::read_package_temp_celsius() as f64;
            let mut c_epsilon = {
                let freq = crate::thermal::read_freq_khz() as f64;
                (freq / max_freq_khz as f64).clamp(0.0, 1.0)
            };
            let mut n_threads = 6;
            // Initial thread count from thermal state
            {
                let t = temp_celsius;
                let ce = c_epsilon;
                if t > 85.0 { n_threads = 1; }
                else if ce > 0.86 { n_threads = 6; }
                else if ce > 0.7 { n_threads = 4; }
                else if t > 78.0 { n_threads = 2; }
                else { n_threads = 6; }
            }

            // Local PolicyEngine for hot-reload inside the blocking thread
            let local_policy = PolicyEngine::new(&policy_path, n_threads)
                .unwrap_or_else(|_| PolicyEngine::new("", n_threads).unwrap());

            // Per-layer profiler (HLC timestamps) — only active if SWAMP_PROFILE=1 or RUST_LOG=debug
            crate::hlc::init_profiling();
            let mut profiler = crate::hlc::ProfileSink::new();

            // Buffers de estado (Forward Pass) locais para a thread bloqueante
            let mut x = vec![0.0f32; embed_dim];
            let mut x_norm = vec![0.0f32; embed_dim];
            let mut q = vec![0.0f32; num_heads * head_dim];
            let mut k = vec![0.0f32; num_kv_heads * head_dim];
            let mut v = vec![0.0f32; num_kv_heads * head_dim];
            let mut attn_out = vec![0.0f32; embed_dim];
            let mut ffn_gate = vec![0.0f32; ffn_dim];
            let mut ffn_up = vec![0.0f32; ffn_dim];
            let mut ffn_down = vec![0.0f32; embed_dim];
            let mut logits = vec![0.0f32; model.config.vocab_size];
            let mut wo_out = vec![0.0f32; embed_dim];
            // Entropy tracker: per-position attention importance for eviction
            let mut importance: Vec<f64> = Vec::new();

            // KV Cache Paginado
            let mut kv_cache = crate::cache::PagedKVCache::new(
                num_layers,
                num_kv_heads,
                model.config.context_len.max(2048),
                head_dim,
            );

            // Fugu orchestrator de estratégias
            let fugu = FuguOrchestrator::new(num_layers, num_heads, num_kv_heads, head_dim);

            // DSPark cognitivo — LSH-based pattern draft + attention candidate selection
            let mut dspark = DSparkEngine::new(embed_dim);
            // Cross-session draft cache: load persisted patterns from NVMe (Idea #3)
            let draft_path = std::env::var("SWAMP_DRAFT_PATH")
                .unwrap_or_else(|_| "/tmp/swamp_draft_cache.bin".to_string());
            dspark.load_draft_cache(&draft_path);
            let mut draft_observations = dspark.draft_model.len;

            // GPU acceleration via GpuComputeContext (runtime-fallback)
            let mut per_layer_gpu: Option<crate::scheduler::PerLayerGpuState> = {
                let window_size = 4096usize.min(model.config.context_len.max(4096));
                let rms_eps = 1e-5_f32;
                let mut pgs = crate::scheduler::PerLayerGpuState::new(
                    window_size, num_heads, num_kv_heads, head_dim, num_layers,
                    embed_dim, ffn_dim, rms_eps,
                    vec![], vec![], vec![], vec![], vec![], vec![], vec![],
                    std::ptr::null_mut(), std::ptr::null_mut(), 0, 0,
                );
                if let Some(ref mut gpu) = pgs {
                    gpu.upload_all_weights(&model);
                    gpu.upload_norm_weights(&attn_norms, &ffn_norms);
                }
                pgs
            };

            // PrefetchEngine for madvise-based page prefetch
            let (mmap_ptr, mmap_len) = model.gguf.mmap_ptr_and_len();
            let prefetch_engine = crate::prefetch::PrefetchEngine::new(mmap_ptr, mmap_len);

            // AIMD Resource Ramp — dobra budget a cada 0.5s sem stress, corta metade no 1o sinal
            let mut aimd = crate::aimd::AimdRamp::new();

            // PowerArbiter — divide budget entre CPU threads e iGPU por RAPL+temp
            let mut arbiter = crate::power_arbiter::PowerArbiter::new();

            // StagingBuffer — workers desacoplados produzem resultado parcial em RAM compartilhada
            // slices = num_heads para attention, slice_size = head_dim
            let num_slices = num_kv_heads.max(1);
            let mut staging = crate::staging::StagingBuffer::new(num_slices, head_dim);

            // VirtualExpertPrefetcher: divide FFN rows em experts, DSPark-guided prefetch
            let num_virtual_experts = 8;
            let mut expert_prefetcher = crate::virtual_experts::VirtualExpertPrefetcher::new(
                num_virtual_experts, ffn_dim,
            );

            let mut tokens = prompt_tokens.clone();

            let mut tokens_generated = 0;
            let t_start = Instant::now();

            // Run in a dedicated Rayon pool
            get_rayon_pool().install(|| -> anyhow::Result<()> {
                // Batched prefill for prompt tokens (skip auto-regressive loop)
                #[allow(unused_mut)]
                let mut prefill_pos = 0;
                if prompt_tokens.len() > 1 {
                    let n_prompt = prompt_tokens.len();
                    let last_x = prefill_batch(
                        num_layers, embed_dim, num_heads, num_kv_heads, head_dim, ffn_dim, rms_eps as f32,
                        &prompt_tokens, &token_embd, &attn_norms, &ffn_norms, &model, &mut kv_cache,
                        #[cfg(feature = "gpu")] &mut per_layer_gpu,
                        n_threads,
                    );
                    // Advance KV cache position past prefill
                    for _ in 0..n_prompt - 1 {
                        kv_cache.advance();
                    }
                    // Final RMSNorm + logits + sample from last prompt token
                    rmsnorm(&mut x_norm, &last_x, &output_norm, rms_eps);
                    forward_linear(
                        &model.gguf, model.gguf.tensor_or_err("output.weight").unwrap(),
                        &x_norm, &mut logits, n_threads,
                    )?;
                    let next_token = sampler.sample(&logits);
                    tokens.push(next_token);
                    // DSPark observe for the last prefill token
                    dspark.observe_at(&last_x, next_token, n_prompt - 1);
                    let next_word = crate::tokenizer::decode(&[next_token]);
                    let _ = tx.blocking_send(next_word);
                    tokens_generated += 1;
                    kv_cache.advance(); // advance past the last prompt token

                    prefill_pos = n_prompt;
                    tracing::info!("Prefill done: {} tokens in {:.2}s", n_prompt, t_start.elapsed().as_secs_f64());
                }

                // Reset entropy tracker for this request
                kv_cache.reset_entropy();

                // DSPark-predicted cold pages for proactive prefetch
                let mut predicted_cold_pages: Vec<usize> = Vec::new();

                let total_steps = req.max_tokens + prompt_tokens.len() - 1;
                for step in prefill_pos..total_steps {
                    // Hot-reload check: try to reload Lua policy every 10 steps
                    if step % 10 == 0 {
                        local_policy.try_reload();
                    }

                    let token_id = if step < prompt_tokens.len() {
                        prompt_tokens[step]
                    } else {
                        *tokens.last().unwrap()
                    };
                    let pos = step;

                    // Embedding lookup
                    let embd_start = token_id * embed_dim;
                    x.copy_from_slice(&token_embd[embd_start..embd_start + embed_dim]);

                    // Proactive DSPark-guided cold page prefetch (before layer loop)
                    for &pid in &predicted_cold_pages {
                        kv_cache.ensure_pages_hot(pid, pid);
                    }
                    predicted_cold_pages.clear();

                    // Thermal telemetry + AIMD resource ramping (every 20 steps ≈ 0.5–1s)
                    if step % 20 == 0 && step > 0 {
                        temp_celsius = crate::thermal::read_package_temp_celsius() as f64;
                        let freq = crate::thermal::read_freq_khz() as f64;
                        c_epsilon = (freq / max_freq_khz as f64).clamp(0.0, 1.0);

                        // Stress signals: temp > warning, freq droop, or RAPL power > 80W
                        let stressed = temp_celsius > 80.0 || c_epsilon < 0.85;
                        let _aimd_budget = aimd.assess(stressed);

                        // PowerArbiter split: CPU vs iGPU
                        let (cpu_frac, _igpu_frac) = arbiter.reassess(55.0);

                        // Scale threads by AIMD budget + PowerArbiter CPU fraction
                        let base_threads = if !stressed { 6 } else { 2 };
                        n_threads = aimd.scaled_threads(base_threads);
                        n_threads = (n_threads as f64 * cpu_frac).round().max(1.0) as usize;
                    }

                    // AIMD stress signal for Fugu (predictive throttle)
                    let predictive_throttle = aimd.budget() < 1.0;

                    // Decide Fugu strategy for this step (before layer loop)
                    let seq_len = kv_cache.current_pos() + 1;
                    let snapshot = SystemSnapshot {
                        seq_len,
                        cache_pressure: 0.0,
                        n_prompt_tokens: prompt_tokens.len(),
                        batch_size: aimd.concurrency(),
                        is_prefill: false,
                        coherence: c_epsilon,
                        dspark_accept_rate: fugu.accept_rate(),
                        predictive_throttle,
                    };
                    let strategy = fugu.decide(&snapshot);
                    let dspark_attn_positions: Vec<usize> = match strategy.attention {
                        AttentionStrategy::SparseWithDSPark { num_dspark, .. } => {
                            dspark.find_attention_candidates(&x, num_dspark)
                                .into_iter().map(|(p, _)| p).collect()
                        }
                        _ => Vec::new(),
                    };

                    // Forward Pass: Camadas (group-aware for weight sharing)
                    // Fused GPU path: async graph replay per layer, no sync until end if all fused
                    #[cfg(feature = "gpu")]
                    let mut fused_all_ok = per_layer_gpu.is_some();
                    #[cfg(not(feature = "gpu"))]
                    let mut fused_all_ok = false;
                    if fused_all_ok {
                        #[cfg(feature = "gpu")]
                        if let Some(ref mut gpu) = per_layer_gpu {
                            gpu.upload_x(&x);
                        }
                    }

                    for group in &model.shared_groups {
                        // Load shared tensor metadata once per group
                        let first = group[0];
                        let q_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_q.weight", first))?;
                        let k_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_k.weight", first))?;
                        let v_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_v.weight", first))?;
                        let o_t = model.gguf.tensor_or_err(&format!("blk.{}.attn_output.weight", first))?;
                        let gate_t = model.gguf.tensor_or_err(&format!("blk.{}.ffn_gate.weight", first))?;
                        let up_t = model.gguf.tensor_or_err(&format!("blk.{}.ffn_up.weight", first))?;
                        let down_t = model.gguf.tensor_or_err(&format!("blk.{}.ffn_down.weight", first))?;

                        for &l in group {
                            // Policy check: skip layer if thermal conditions require it
                            if local_policy.should_skip_layer(l, temp_celsius) {
                                continue;
                            }

                            #[cfg(feature = "gpu")]
                            let mut layer_profile = profiler.begin_layer(l, per_layer_gpu.is_some());
                            #[cfg(not(feature = "gpu"))]
                            let mut layer_profile = profiler.begin_layer(l, false);

                            // Fused GPU layer graph: async replay on stream (no sync unless CPU fallback)
                            #[cfg(feature = "gpu")]
                            let layer_fused_ok: bool = if fused_all_ok {
                                let ok = per_layer_gpu.as_mut().map_or(false, |gpu| {
                                    let _ = gpu.create_layer_graph(l, embed_dim, ffn_dim);
                                    gpu.execute_layer_fused(l, pos)
                                });
                                if !ok { fused_all_ok = false; }
                                ok
                            } else { false };
                            #[cfg(not(feature = "gpu"))]
                            let layer_fused_ok = false;
                            if layer_fused_ok {
                                profiler.end_layer(layer_profile);
                                continue;
                            }
                            // First CPU-fallback layer: sync compute stream and download d_x to host x
                            #[cfg(feature = "gpu")]
                            if !fused_all_ok {
                                if let Some(ref mut gpu) = per_layer_gpu {
                                    gpu.sync();
                                    gpu.download_x(&mut x);
                                }
                                fused_all_ok = false;
                            }

                            // RMSNorm Attn
                            rmsnorm(&mut x_norm, &x, &attn_norms[l], rms_eps);

                            // QKV Projections FUNDIDAS (shared weight across group)
                            let gemv_qkv_ok: bool = {
                                #[cfg(feature = "gpu")]
                                {
                                    per_layer_gpu.as_mut().map_or(false, |gpu| {
                                        let n_blocks = q_t.shape[0] / 256;
                                        let ok = gpu.gemv_qkv_async(
                                            l,
                                            std::ptr::null_mut(),
                                            std::ptr::null_mut(),
                                            std::ptr::null_mut(),
                                            &x_norm, &mut q, &mut k, &mut v,
                                            q_t.shape[1] as i32,
                                            k_t.shape[1] as i32,
                                            v_t.shape[1] as i32,
                                            n_blocks as i32,
                                        );
                                        ok && gpu.sync()
                                    })
                                }
                                #[cfg(not(feature = "gpu"))]
                                { false }
                            };
                            if !gemv_qkv_ok {
                                forward_linear_multi(
                                    &model.gguf,
                                    &[q_t, k_t, v_t],
                                    &x_norm,
                                    &mut [&mut q, &mut k, &mut v],
                                    n_threads,
                                )?;
                            }

                            // RoPE
                            apply_rope_ufc(&mut q, &mut k, pos, num_heads, num_kv_heads, head_dim, model.config.context_len);

                            // KV Cache (CPU)
                            kv_cache.save(l, &k, &v);
                            let seq_len = kv_cache.current_pos() + 1;

                            // Pre-heat KV pages from cold storage before parallel attention
                    kv_cache.ensure_pages_hot(0, kv_cache.page_id(seq_len.saturating_sub(1)));

                            // Attention dispatch via Fugu strategy
                            let attn_dispatched: bool = {
                                #[cfg(feature = "gpu")]
                                {
                                    let gpu_ok = per_layer_gpu.as_mut().map_or(false, |gpu| {
                                        let gpu_tic = profiler.clock().now();
                                        gpu.upload_kv_async(&k, &v, pos);
                                        let launched = gpu.execute_attention_async(
                                            &q, &mut attn_out, pos, seq_len,
                                        );
                                        if !launched {
                                            return false;
                                        }
                                        let synced = gpu.sync();
                                        let gpu_toc = profiler.clock().now();
                                        if synced {
                                            let gpu_ms = (gpu_toc.wall.saturating_sub(gpu_tic.wall)) as f64 / 1_000_000.0;
                                            profiler.record_gpu_attention(&mut layer_profile, gpu_ms);
                                        }
                                        synced
                                    });
                                    if gpu_ok {
                                        true
                                    } else {
                                        false // fall through to CPU dispatch
                                    }
                                }
                                #[cfg(not(feature = "gpu"))]
                                { false }
                            };

                            if !attn_dispatched {
                                // CPU attention dispatch by Fugu strategy
                                match strategy.attention {
                                    AttentionStrategy::Full => {
                                        importance.resize(seq_len, 0.0);
                                        importance[..seq_len].fill(0.0);
                                        crate::ops::attention_tracked(
                                            &mut attn_out, &q, &mut kv_cache, l, seq_len, pos,
                                            num_heads, num_kv_heads, head_dim,
                                            &mut importance[..seq_len],
                                        );
                                        // Propagate entropy weights to cache pages
                                        for (t, &w) in importance[..seq_len].iter().enumerate() {
                                            if w > 0.001 {
                                                kv_cache.record_importance(t, w);
                                            }
                                        }
                                    }
                                    AttentionStrategy::Sparse { window, .. } => {
                                        crate::ops::attention_sparse(&mut attn_out, &q, &mut kv_cache, l, seq_len, pos, num_heads, num_kv_heads, head_dim, window, 64);
                                    }
                                    AttentionStrategy::SparseWithDSPark { window, sentinel_stride, .. } => {
                                        crate::ops::attention_fugu(&mut attn_out, &q, &mut kv_cache, l, seq_len, pos, num_heads, num_kv_heads, head_dim, window, sentinel_stride, &dspark_attn_positions);
                                    }
                                }
                            }

                            // Output Projection (shared weight)
                            let gemv_o_ok: bool = {
                                #[cfg(feature = "gpu")]
                                {
                                    per_layer_gpu.as_mut().map_or(false, |gpu| {
                                        let ok = gpu.execute_gemv_async(
                                            l,
                                            std::ptr::null_mut(),
                                            &attn_out, &mut wo_out,
                                            o_t.shape[1] as i32,
                                            (o_t.shape[0] / 256) as i32,
                                        );
                                        ok && gpu.sync()
                                    })
                                }
                                #[cfg(not(feature = "gpu"))]
                                { false }
                            };
                            if !gemv_o_ok {
                                forward_linear(&model.gguf, o_t, &attn_out, &mut wo_out, n_threads)?;
                            }
                            add_in_place(&mut x, &wo_out);

                            // RMSNorm FFN
                            rmsnorm(&mut x_norm, &x, &ffn_norms[l], rms_eps);

                            // FFN Gate & Up FUNDIDOS (shared weight)
                            let gemv_gu_ok: bool = {
                                #[cfg(feature = "gpu")]
                                {
                                    per_layer_gpu.as_mut().map_or(false, |gpu| {
                                        let n_blocks = gate_t.shape[0] / 256;
                                        let ok = gpu.gemv_gate_up_async(
                                            l,
                                            std::ptr::null_mut(),
                                            std::ptr::null_mut(),
                                            &x_norm, &mut ffn_gate, &mut ffn_up,
                                            gate_t.shape[1] as i32,
                                            n_blocks as i32,
                                        );
                                        ok && gpu.sync()
                                    })
                                }
                                #[cfg(not(feature = "gpu"))]
                                { false }
                            };
                            if !gemv_gu_ok {
                                forward_linear_multi(
                                    &model.gguf,
                                    &[gate_t, up_t],
                                    &x_norm,
                                    &mut [&mut ffn_gate, &mut ffn_up],
                                    n_threads,
                                )?;
                            }

                            silu(&mut ffn_gate);
                            mul_in_place(&mut ffn_gate, &ffn_up);

                            // FFN Down (shared weight)
                            let gemv_down_ok: bool = {
                                #[cfg(feature = "gpu")]
                                {
                                    per_layer_gpu.as_mut().map_or(false, |gpu| {
                                        let ok = gpu.execute_gemv_async(
                                            l,
                                            std::ptr::null_mut(),
                                            &ffn_gate, &mut ffn_down,
                                            down_t.shape[1] as i32,
                                            (down_t.shape[0] / 256) as i32,
                                        );
                                        ok && gpu.sync()
                                    })
                                }
                                #[cfg(not(feature = "gpu"))]
                                { false }
                            };
                            if !gemv_down_ok {
                                forward_linear(&model.gguf, down_t, &ffn_gate, &mut ffn_down, n_threads)?;
                            }
                            add_in_place(&mut x, &ffn_down);

                            // Prefetch next group's first layer tensors (madvise WILLNEED)
                            // Only prefetch if next layer is in a different group (different weights)
                            let is_last_in_group = l == *group.last().unwrap();
                            if is_last_in_group {
                                let next = l + 1;
                                if next < num_layers {
                                    for tensor_name in &[
                                        format!("blk.{}.attn_q.weight", next),
                                        format!("blk.{}.attn_k.weight", next),
                                        format!("blk.{}.attn_v.weight", next),
                                        format!("blk.{}.attn_output.weight", next),
                                        format!("blk.{}.ffn_gate.weight", next),
                                        format!("blk.{}.ffn_up.weight", next),
                                        format!("blk.{}.ffn_down.weight", next),
                                    ] {
                                        if let Some((off, len)) = model.gguf.tensor_raw_offset_len(tensor_name) {
                                            prefetch_engine.prefetch_range(off, len);
                                        }
                                    }
                                }
                            }

                            profiler.end_layer(layer_profile);
                        }
                    }

                    // If all layers used fused graph: sync + download final x to host
                    #[cfg(feature = "gpu")]
                    if fused_all_ok {
                        if let Some(ref mut gpu) = per_layer_gpu {
                            gpu.sync();
                            gpu.download_x(&mut x);
                        }
                    }

                    // RMSNorm Final
                    rmsnorm(&mut x_norm, &x, &output_norm, rms_eps);

                    // Logits
                    forward_linear(&model.gguf, model.gguf.tensor_or_err("output.weight")?, &x_norm, &mut logits, n_threads)?;

                    // Amostragem (só depois de preencher o cache com o prompt)
                    if step >= prompt_tokens.len() - 1 {
                        let mut next_token = sampler.sample(&logits);

                        // Speculative decoding: DSPark draft → reject if matches sampled token
                        let draft_result = dspark.draft_model.draft(&x);
                        let accepted_draft = if !draft_result.0.is_empty()
                            && draft_result.1[0] > 0.6
                            && draft_result.0[0] == next_token
                        {
                            // Draft matches sampled token — accept it (saved forward pass)
                            Some(draft_result.0[0])
                        } else {
                            None
                        };

                        if accepted_draft.is_none() {
                            tokens.push(next_token);
                            dspark.observe_at(&x, next_token, pos);
                        } else {
                            tokens.push(draft_result.0[0]);
                            dspark.observe_at(&x, draft_result.0[0], pos);
                        }
                        // Persist draft cache every 100 observations (Idea #3)
                        draft_observations += 1;
                        dspark.draft_model.save_if_due(&draft_path, draft_observations);

                        // MoE Expert Prefetch: only when AIMD budget is healthy (not stressed)
                        if aimd.budget() >= 0.5 {
                            let lsh_hash = dspark.draft_model.hash_hidden(&x);
                            let (start_row, end_row) = expert_prefetcher.predict_expert(lsh_hash);
                            let num_expert_rows = end_row - start_row;
                            if num_expert_rows > 0 {
                                for l in 0..num_layers {
                                    for name in &[
                                        format!("blk.{}.ffn_gate.weight", l),
                                        format!("blk.{}.ffn_up.weight", l),
                                    ] {
                                        if let Some((off, len)) = model.gguf.tensor_raw_offset_len(name) {
                                            let row_size = len / ffn_dim;
                                            let e_off = off + start_row * row_size;
                                            let e_len = num_expert_rows * row_size;
                                            prefetch_engine.prefetch_range(e_off, e_len);
                                        }
                                    }
                                }
                            }
                        }

                        // DSPark-guided cold page prefetch: predict which KV pages
                        // will be accessed in upcoming steps and start loading them
                        let next_pages = dspark.predict_cold_pages(
                            &x, pos, 3, 4096, 4096,
                            |p| kv_cache.page_id(p),
                        );
                        predicted_cold_pages.extend(next_pages);
                        predicted_cold_pages.sort();
                        predicted_cold_pages.dedup();
                        predicted_cold_pages.truncate(64);

                        let next_word = crate::tokenizer::decode(&[next_token]);
                        
                        if let Err(_) = tx.blocking_send(next_word) {
                            break; // Canal fechado
                        }

                        tokens_generated += 1;
                    }
                    
                    kv_cache.advance();

                    // Reset staging buffer for next token
                    staging.reset_all();
                }
                
                let elapsed = t_start.elapsed().as_secs_f64();

                // Profiler summary
                if !profiler.is_empty() {
                    let _ = tx.blocking_send(format!("\n—— HLC Profile ——"));
                    let report = profiler.report();
                    let _ = tx.blocking_send(report);
                }

                if elapsed > 0.0 {
                    let _ = tx.blocking_send(format!("\n\n[Tokens: {} | Throughput: {:.2} tok/s]", tokens_generated, tokens_generated as f64 / elapsed));
                }
                Ok(())
            })
        }).await;

        match result {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(e) => Err(anyhow::anyhow!("Join Error: {:?}", e)),
        }
    }

    /// Pipeline concurrente: processa até `max_concurrency` requests em paralelo,
    /// sobrepondo prefill de uma com decode de outra via pipeline parallelism.
    /// Cada request roda em sua própria thread com KV cache independente.
    pub fn generate_batch(
        &self,
        requests: Vec<InferenceRequest>,
        max_concurrency: usize,
    ) -> Vec<anyhow::Result<String>> {
        use std::sync::{mpsc, Arc, Mutex};
        use std::thread;

        let n = requests.len();
        if n == 0 { return Vec::new(); }

        let results: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(vec![None; n]));
        let errors: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(vec![None; n]));

        // Use a shared queue with mutex
        let job_queue: Arc<Mutex<Vec<(usize, InferenceRequest)>>> = Arc::new(Mutex::new(
            requests.into_iter().enumerate().collect()
        ));

        let mut handles = Vec::with_capacity(max_concurrency);
        for _ in 0..max_concurrency.min(n) {
            let jq = job_queue.clone();
            let res = results.clone();
            let errs = errors.clone();
            let model = self.model.clone();
            let token_embd = self.token_embd.clone();
            let attn_norms = self.attn_norms.clone();
            let ffn_norms = self.ffn_norms.clone();
            let output_norm = self.output_norm.clone();
            let policy_script = self.policy.script_path().to_string();

            handles.push(thread::spawn(move || {
                loop {
                    let job = {
                        let mut q = jq.lock().unwrap();
                        q.pop()
                    };
                    let (idx, req) = match job {
                        Some(j) => j,
                        None => break,
                    };
                    let executor = ModelExecutor {
                        model: model.clone(),
                        thermal_coordinator: ThermalCoordinator::default(),
                        prefetcher: LscPrefetcher::default(),
                        policy: PolicyEngine::new(&policy_script, 6).unwrap_or_else(|_| PolicyEngine::new("", 6).unwrap()),
                        token_embd: token_embd.clone(),
                        attn_norms: attn_norms.clone(),
                        ffn_norms: ffn_norms.clone(),
                        output_norm: output_norm.clone(),
                    };
                    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
                    let result = tokio::runtime::Runtime::new()
                        .unwrap()
                        .block_on(async move {
                            let mut exec = executor;
                            let _ = exec.generate(req, tx.clone()).await;
                            let mut output = String::new();
                            while let Some(msg) = rx.blocking_recv() {
                                output.push_str(&msg);
                            }
                            output
                        });
                    let mut r = res.lock().unwrap();
                    r[idx] = Some(result);
                }
            }));
        }

        for h in handles {
            let _ = h.join();
        }

        let final_results = results.lock().unwrap();
        final_results.iter().map(|r| {
            match r {
                Some(s) => Ok(s.clone()),
                None => Err(anyhow::anyhow!("request failed")),
            }
        }).collect()
    }
}
