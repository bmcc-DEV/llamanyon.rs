// swamp-engine/src/swamp_vm.rs
// SwampVM — dispatcher persistente de opcodes de passo de modelo.
//
// Mesmo padrão do FVM (fila de opcodes + dispatcher contínuo), subido
// do nível de "GEMV dentro de uma camada" pra "passo de modelo entre sessões".

#![allow(unused)]

use crate::governor::{ResourceGovernor, WorkloadClass, SessionId, CognitiveWorkerPool};
use crate::model::Model;
use crate::model_registry::MemoryTier;
use crate::linear::{forward_linear};
use crate::ops::{rmsnorm, silu, add_in_place, mul_in_place, apply_rope_ufc};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Opcode
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct ModelStepOpcode {
    pub session_id: SessionId,
    pub generation: u64,
    pub model_id: u32,
    pub step_type: StepType,
    pub workload_class: WorkloadClass,
    pub ctx: Arc<std::sync::Mutex<SessionContext>>,
    pub token_id: u32,
    pub pos: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepType {
    PrefillChunk,
    DecodeToken,
}

// ---------------------------------------------------------------------------
// SessionContext — buffers reais + KV cache da sessão
// ---------------------------------------------------------------------------

pub struct SessionContext {
    // config (cached do model)
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub rms_eps: f32,
    pub context_len: usize,
    pub max_seq_len: usize,

    // buffers — acesso exclusivo pelo dispatcher
    pub x: Vec<f32>,
    pub x_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub wo_out: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
    pub logits: Vec<f32>,

    pub kv_cache: crate::cache::PagedKVCache,
    pub result_tx: Option<tokio::sync::oneshot::Sender<String>>,
}

impl SessionContext {
    pub fn new(
        embed_dim: usize, num_heads: usize, num_kv_heads: usize,
        head_dim: usize, ffn_dim: usize, num_layers: usize,
        vocab_size: usize, context_len: usize, rms_eps: f32,
    ) -> Self {
        let max_seq_len = context_len.max(2048);
        Self {
            embed_dim, num_heads, num_kv_heads, head_dim,
            ffn_dim, num_layers, vocab_size, rms_eps, context_len,
            max_seq_len,
            x: vec![0.0; embed_dim],
            x_norm: vec![0.0; embed_dim],
            q: vec![0.0; num_heads * head_dim],
            k: vec![0.0; num_kv_heads * head_dim],
            v: vec![0.0; num_kv_heads * head_dim],
            attn_out: vec![0.0; embed_dim],
            wo_out: vec![0.0; embed_dim],
            ffn_gate: vec![0.0; ffn_dim],
            ffn_up: vec![0.0; ffn_dim],
            ffn_down: vec![0.0; embed_dim],
            logits: vec![0.0; vocab_size],
            kv_cache: crate::cache::PagedKVCache::new(
                num_layers, num_kv_heads, max_seq_len, head_dim,
            ),
            result_tx: None,
        }
    }
}

// ---------------------------------------------------------------------------
// ModelTable
// ---------------------------------------------------------------------------

struct ModelSlot {
    model: Arc<Model>,
    model_name: String,
    model_size: usize,
    tier: MemoryTier,
    gpu_state: Option<Box<crate::scheduler::PerLayerGpuState>>,
    // Cached norms (carregados uma vez no register_model)
    attn_norms: Vec<Vec<f32>>,
    ffn_norms: Vec<Vec<f32>>,
    output_norm: Vec<f32>,
    token_embd: Vec<f32>,
}

impl ModelSlot {
    fn can_route_to_gpu(&self, threshold: usize) -> bool {
        self.model_size >= threshold
            && self.tier == MemoryTier::Vram
            && self.gpu_state.is_some()
    }
}

// ---------------------------------------------------------------------------
// SwampVm
// ---------------------------------------------------------------------------

pub struct SwampVm {
    ring: crossbeam::channel::Receiver<ModelStepOpcode>,
    ring_tx: crossbeam::channel::Sender<ModelStepOpcode>,
    models: Vec<ModelSlot>,
    governor: Arc<std::sync::Mutex<ResourceGovernor>>,
    cognitive_pool: CognitiveWorkerPool,
    stats: VmStats,
}

#[derive(Debug, Clone, Default)]
pub struct VmStats {
    pub total_ops: u64,
    pub gpu_ops: u64,
    pub cpu_ops: u64,
    pub cognitive_ops: u64,
    pub stale_dropped: u64,
}

impl SwampVm {
    pub fn new(governor: Arc<std::sync::Mutex<ResourceGovernor>>) -> Self {
        let (tx, rx) = crossbeam::channel::unbounded();
        Self {
            ring: rx,
            ring_tx: tx,
            models: Vec::new(),
            governor,
            cognitive_pool: CognitiveWorkerPool::new(12),
            stats: VmStats::default(),
        }
    }

    pub fn register_model(&mut self, name: &str, model: Arc<Model>, tier: MemoryTier) {
        let size = model.gguf.mmap_ptr_and_len().1;
        let num_layers = model.config.num_layers;
        let mut attn_norms = Vec::with_capacity(num_layers);
        let mut ffn_norms = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let an = model.gguf.dequantize_tensor_alloc(
                model.gguf.tensor_or_err(&format!("blk.{l}.attn_norm.weight")).unwrap()
            ).unwrap();
            let fn_ = model.gguf.dequantize_tensor_alloc(
                model.gguf.tensor_or_err(&format!("blk.{l}.ffn_norm.weight")).unwrap()
            ).unwrap();
            attn_norms.push(an);
            ffn_norms.push(fn_);
        }
        let output_norm = model.gguf.dequantize_tensor_alloc(
            model.gguf.tensor_or_err("output_norm.weight").unwrap()
        ).unwrap();
        let token_embd = model.gguf.dequantize_tensor_alloc(
            model.gguf.tensor_or_err("token_embd.weight").unwrap()
        ).unwrap();

        let slot = ModelSlot {
            model,
            model_name: name.to_string(),
            model_size: size,
            tier,
            gpu_state: None,
            attn_norms,
            ffn_norms,
            output_norm,
            token_embd,
        };
        self.models.push(slot);
    }

    pub fn enqueue(&self, op: ModelStepOpcode) -> Result<(), crossbeam::channel::TrySendError<ModelStepOpcode>> {
        self.ring_tx.try_send(op)
    }

    pub fn ring_tx(&self) -> crossbeam::channel::Sender<ModelStepOpcode> {
        self.ring_tx.clone()
    }

    // =====================================================================
    // Dispatcher loop (roda numa thread persistente)
    // =====================================================================

    pub fn run_dispatch_loop(vm: Arc<std::sync::Mutex<Self>>) {
        let receiver = {
            let guard = vm.lock().unwrap();
            guard.ring.clone()
        };
        loop {
            match receiver.recv() {
                Ok(op) => {
                    let mut guard = vm.lock().unwrap();
                    guard.dispatch(op);
                }
                Err(crossbeam::channel::RecvError) => break,
            }
        }
    }

    fn dispatch(&mut self, op: ModelStepOpcode) {
        self.stats.total_ops += 1;

        // Stale detection
        let gen = self.governor.lock().unwrap().session_generation(op.session_id);
        if gen.map_or(true, |g| g != op.generation) {
            self.stats.stale_dropped += 1;
            return;
        }

        let slot_idx = op.model_id as usize;
        if slot_idx >= self.models.len() {
            return;
        }

        match op.workload_class {
            WorkloadClass::Cognitive => {
                self.stats.cognitive_ops += 1;
                let slot = &self.models[slot_idx];
                if let Some(_token) = self.cognitive_pool.try_acquire() {
                    let ctx = op.ctx.clone();
                    let model = slot.model.clone();
                    crate::executor::get_rayon_pool().install(move || {
                        Self::execute_cognitive(&ctx, &model);
                    });
                }
            }
            _ => {
                let slot = &mut self.models[slot_idx];
                let threshold = self.governor.lock().unwrap().gpu_model_threshold;
                if slot.can_route_to_gpu(threshold) {
                    self.stats.gpu_ops += 1;
                    Self::execute_gpu(&op, slot);
                } else {
                    self.stats.cpu_ops += 1;
                    Self::execute_cpu(&op, slot);
                }
            }
        }
    }

    /// Inicializa GPU state lazy
    fn init_gpu_state(slot: &mut ModelSlot) {
        if slot.gpu_state.is_some() {
            return;
        }
        let model = &slot.model;
        let cfg = &model.config;
        let head_dim = cfg.embed_dim / cfg.num_heads;
        let ffn_dim = model.gguf.tensor_or_err("blk.0.ffn_gate.weight")
            .map(|t| t.shape[1] as usize).unwrap_or(cfg.embed_dim * 4);
        let state = crate::scheduler::PerLayerGpuState::new(
            4096, cfg.num_heads, cfg.num_kv_heads, head_dim, cfg.num_layers,
            cfg.embed_dim, ffn_dim, 1e-5_f32,
            vec![], vec![], vec![], vec![], vec![], vec![], vec![],
            std::ptr::null_mut(), std::ptr::null_mut(),
            0, 0,
        );
        slot.gpu_state = state.map(Box::new);
    }

    // =====================================================================
    // execute_cpu — layer loop completo via CPU
    // =====================================================================

    fn execute_cpu(op: &ModelStepOpcode, slot: &mut ModelSlot) {
        let model = &slot.model;
        let gguf = &model.gguf;
        let cfg = &model.config;
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.embed_dim / num_heads;
        let rms_eps = 1e-5;
        let pos = op.pos as usize;

        let mut ctx_guard = op.ctx.lock().unwrap();
        let embed_dim = ctx_guard.embed_dim;

        let token_id = op.token_id as usize;
        ctx_guard.x.copy_from_slice(
            &slot.token_embd[token_id * embed_dim..(token_id + 1) * embed_dim]
        );

        let n_threads = 6;
        let mut ctx: &mut SessionContext = &mut *ctx_guard;

        for group in &model.shared_groups {
            let first = group[0];
            let q_t = gguf.tensor_or_err(&format!("blk.{}.attn_q.weight", first)).unwrap();
            let k_t = gguf.tensor_or_err(&format!("blk.{}.attn_k.weight", first)).unwrap();
            let v_t = gguf.tensor_or_err(&format!("blk.{}.attn_v.weight", first)).unwrap();
            let o_t = gguf.tensor_or_err(&format!("blk.{}.attn_output.weight", first)).unwrap();
            let gate_t = gguf.tensor_or_err(&format!("blk.{}.ffn_gate.weight", first)).unwrap();
            let up_t = gguf.tensor_or_err(&format!("blk.{}.ffn_up.weight", first)).unwrap();
            let down_t = gguf.tensor_or_err(&format!("blk.{}.ffn_down.weight", first)).unwrap();

            for &l in group {
                rmsnorm(&mut ctx.x_norm, &ctx.x, &slot.attn_norms[l], rms_eps);

                {
                    let _ = forward_linear(gguf, q_t, &ctx.x_norm, &mut ctx.q, n_threads);
                    let _ = forward_linear(gguf, k_t, &ctx.x_norm, &mut ctx.k, n_threads);
                    let _ = forward_linear(gguf, v_t, &ctx.x_norm, &mut ctx.v, n_threads);
                }

                apply_rope_ufc(&mut ctx.q, &mut ctx.k, pos, num_heads, num_kv_heads, head_dim, cfg.context_len);
                ctx.kv_cache.save(l, &ctx.k, &ctx.v);
                let seq_len = ctx.kv_cache.current_pos() + 1;
                ctx.kv_cache.ensure_pages_hot(0, ctx.kv_cache.page_id(seq_len.saturating_sub(1)));

                crate::ops::attention(&mut ctx.attn_out, &ctx.q, &mut ctx.kv_cache, l, seq_len, pos,
                    num_heads, num_kv_heads, head_dim);

                {
                    let _ = forward_linear(gguf, o_t, &ctx.attn_out, &mut ctx.wo_out, n_threads);
                }
                add_in_place(&mut ctx.x, &ctx.wo_out);

                rmsnorm(&mut ctx.x_norm, &ctx.x, &slot.ffn_norms[l], rms_eps);

                {
                    let _ = forward_linear(gguf, gate_t, &ctx.x_norm, &mut ctx.ffn_gate, n_threads);
                    let _ = forward_linear(gguf, up_t, &ctx.x_norm, &mut ctx.ffn_up, n_threads);
                }

                silu(&mut ctx.ffn_gate);
                mul_in_place(&mut ctx.ffn_gate, &ctx.ffn_up);

                {
                    let _ = forward_linear(gguf, down_t, &ctx.ffn_gate, &mut ctx.ffn_down, n_threads);
                }
                add_in_place(&mut ctx.x, &ctx.ffn_down);
            }
        }

        rmsnorm(&mut ctx.x_norm, &ctx.x, &slot.output_norm, rms_eps);
        let _ = forward_linear(gguf, gguf.tensor_or_err("output.weight").unwrap(),
            &ctx.x_norm, &mut ctx.logits, n_threads);

        use crate::sampler::Sampler;
        let sampler = Sampler::new(0.0, 1, 1.0);
        let next_token = sampler.sample(&ctx.logits);
        let word = crate::tokenizer::decode(&[next_token]);

        if let Some(tx) = ctx.result_tx.take() {
            let _ = tx.send(word);
        }
        ctx.kv_cache.advance();
    }

    // =====================================================================
    // execute_gpu — layer loop com GPU GEMV dispatch
    // =====================================================================

    fn execute_gpu(op: &ModelStepOpcode, slot: &mut ModelSlot) {
        Self::init_gpu_state(slot);
        let gpu = match slot.gpu_state.as_mut() {
            Some(g) => g.as_mut(),
            None => return Self::execute_cpu(op, slot),
        };

        let cfg = &slot.model.config;
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.embed_dim / num_heads;
        let rms_eps = 1e-5;
        let pos = op.pos as usize;

        let mut ctx_guard = op.ctx.lock().unwrap();
        let embed_dim = ctx_guard.embed_dim;
        let token_id = op.token_id as usize;
        ctx_guard.x.copy_from_slice(
            &slot.token_embd[token_id * embed_dim..(token_id + 1) * embed_dim]
        );
        let ctx: &mut SessionContext = &mut *ctx_guard;
        let gguf = &slot.model.gguf;

        for group in &slot.model.shared_groups {
            let first = group[0];
            let q_t = gguf.tensor_or_err(&format!("blk.{}.attn_q.weight", first)).unwrap();
            let k_t = gguf.tensor_or_err(&format!("blk.{}.attn_k.weight", first)).unwrap();
            let v_t = gguf.tensor_or_err(&format!("blk.{}.attn_v.weight", first)).unwrap();
            let o_t = gguf.tensor_or_err(&format!("blk.{}.attn_output.weight", first)).unwrap();
            let gate_t = gguf.tensor_or_err(&format!("blk.{}.ffn_gate.weight", first)).unwrap();
            let up_t = gguf.tensor_or_err(&format!("blk.{}.ffn_up.weight", first)).unwrap();
            let down_t = gguf.tensor_or_err(&format!("blk.{}.ffn_down.weight", first)).unwrap();

            for &l in group {
                rmsnorm(&mut ctx.x_norm, &ctx.x, &slot.attn_norms[l], rms_eps);

                let n_blocks = q_t.shape[0] / 256;
                let qkv_ok = gpu.gemv_qkv_async(l,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &ctx.x_norm, &mut ctx.q, &mut ctx.k, &mut ctx.v,
                    q_t.shape[1] as i32, k_t.shape[1] as i32, v_t.shape[1] as i32,
                    n_blocks as i32,
                ) && gpu.sync();
                if !qkv_ok {
                    let _ = forward_linear(gguf, q_t, &ctx.x_norm, &mut ctx.q, 6);
                    let _ = forward_linear(gguf, k_t, &ctx.x_norm, &mut ctx.k, 6);
                    let _ = forward_linear(gguf, v_t, &ctx.x_norm, &mut ctx.v, 6);
                }

                apply_rope_ufc(&mut ctx.q, &mut ctx.k, pos, num_heads, num_kv_heads, head_dim, cfg.context_len);
                ctx.kv_cache.save(l, &ctx.k, &ctx.v);
                let seq_len = ctx.kv_cache.current_pos() + 1;
                ctx.kv_cache.ensure_pages_hot(0, ctx.kv_cache.page_id(seq_len.saturating_sub(1)));

                gpu.upload_kv_async(&ctx.k, &ctx.v, pos);
                let attn_ok = gpu.execute_attention_async(&ctx.q, &mut ctx.attn_out, pos, seq_len)
                    && gpu.sync();
                if !attn_ok {
                    crate::ops::attention(&mut ctx.attn_out, &ctx.q, &mut ctx.kv_cache, l, seq_len, pos,
                        num_heads, num_kv_heads, head_dim);
                }

                let o_ok = gpu.execute_gemv_async(l,
                    std::ptr::null_mut(),
                    &ctx.attn_out, &mut ctx.wo_out,
                    o_t.shape[1] as i32, (o_t.shape[0] / 256) as i32,
                ) && gpu.sync();
                if !o_ok {
                    let _ = forward_linear(gguf, o_t, &ctx.attn_out, &mut ctx.wo_out, 6);
                }
                add_in_place(&mut ctx.x, &ctx.wo_out);

                rmsnorm(&mut ctx.x_norm, &ctx.x, &slot.ffn_norms[l], rms_eps);

                let n_blocks_gu = gate_t.shape[0] / 256;
                let gu_ok = gpu.gemv_gate_up_async(l,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &ctx.x_norm, &mut ctx.ffn_gate, &mut ctx.ffn_up,
                    gate_t.shape[1] as i32, n_blocks_gu as i32,
                ) && gpu.sync();
                if !gu_ok {
                    let _ = forward_linear(gguf, gate_t, &ctx.x_norm, &mut ctx.ffn_gate, 6);
                    let _ = forward_linear(gguf, up_t, &ctx.x_norm, &mut ctx.ffn_up, 6);
                }

                silu(&mut ctx.ffn_gate);
                mul_in_place(&mut ctx.ffn_gate, &ctx.ffn_up);

                let down_ok = gpu.execute_gemv_async(l,
                    std::ptr::null_mut(),
                    &ctx.ffn_gate, &mut ctx.ffn_down,
                    down_t.shape[1] as i32, (down_t.shape[0] / 256) as i32,
                ) && gpu.sync();
                if !down_ok {
                    let _ = forward_linear(gguf, down_t, &ctx.ffn_gate, &mut ctx.ffn_down, 6);
                }
                add_in_place(&mut ctx.x, &ctx.ffn_down);
            }
        }

        rmsnorm(&mut ctx.x_norm, &ctx.x, &slot.output_norm, rms_eps);
        let _ = forward_linear(gguf, gguf.tensor_or_err("output.weight").unwrap(),
            &ctx.x_norm, &mut ctx.logits, 6);

        use crate::sampler::Sampler;
        let sampler = Sampler::new(0.0, 1, 1.0);
        let next_token = sampler.sample(&ctx.logits);
        let word = crate::tokenizer::decode(&[next_token]);

        if let Some(tx) = ctx.result_tx.take() {
            let _ = tx.send(word);
        }
        ctx.kv_cache.advance();
    }

    // =====================================================================
    // execute_cognitive — classificadores CPU-only
    // =====================================================================

    fn execute_cognitive(ctx: &std::sync::Mutex<SessionContext>, model: &Model) {
        let mut c = ctx.lock().unwrap();
        let embed_dim = c.embed_dim;
        let token_embd = model.gguf.dequantize_tensor_alloc(
            model.gguf.tensor_or_err("token_embd.weight").unwrap()
        ).unwrap();
        c.x.copy_from_slice(&token_embd[0..embed_dim]);
    }

    pub fn stats(&self) -> &VmStats { &self.stats }
    pub fn reset_stats(&mut self) { self.stats = VmStats::default(); }
}
