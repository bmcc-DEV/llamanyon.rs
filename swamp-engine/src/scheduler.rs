use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use ash::vk;

// ---------------------------------------------------------------------------
// PerLayerGpuState: orchestrates one layer's GPU compute
// Wraps GpuComputeContext with the same interface executor.rs expects
// Falls back gracefully (returns false) if GPU is unavailable
// ---------------------------------------------------------------------------

pub struct PerLayerGpuState {
    ctx: Option<swamp_gpu::GpuComputeContext>,
    pub device: Option<Arc<swamp_gpu::GpuDevice>>,
    enabled: bool,
    num_layers: usize,
    window_size: usize,
    num_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    ffn_dim: usize,
}

unsafe impl Send for PerLayerGpuState {}
unsafe impl Sync for PerLayerGpuState {}

impl PerLayerGpuState {
    pub fn new(
        window_size: usize, num_heads: usize, n_kv_heads: usize, head_dim: usize,
        num_layers: usize, embed_dim: usize, ffn_dim: usize,
        _rms_eps: f32,
        _d_q_weight: Vec<*mut u8>, _d_k_weight: Vec<*mut u8>, _d_v_weight: Vec<*mut u8>,
        _d_o_weight: Vec<*mut u8>, _d_gate_weight: Vec<*mut u8>, _d_up_weight: Vec<*mut u8>,
        _d_down_weight: Vec<*mut u8>,
        _d_gemv_x: *mut f32, _d_gemv_out: *mut f32,
        _max_gemv_cols: i32, _max_gemv_rows: i32,
    ) -> Option<Self> {
        let device = Arc::new(swamp_gpu::GpuDevice::new());
        if !device.enabled {
            return Some(Self { ctx: None, device: Some(device), enabled: false,
                num_layers, window_size, num_heads, n_kv_heads, head_dim, embed_dim, ffn_dim });
        }
        let ctx = swamp_gpu::GpuComputeContext::new(
            &device, embed_dim, ffn_dim, num_heads, n_kv_heads, head_dim, window_size, num_layers,
        );
        if ctx.is_none() {
            return Some(Self { ctx: None, device: Some(device), enabled: false,
                num_layers, window_size, num_heads, n_kv_heads, head_dim, embed_dim, ffn_dim });
        }
        Some(Self { ctx, device: Some(device), enabled: true,
            num_layers, window_size, num_heads, n_kv_heads, head_dim, embed_dim, ffn_dim })
    }

    pub fn is_operational(&self) -> bool {
        self.enabled && self.ctx.is_some()
    }

    pub fn upload_weight(_h_w: *const u8, _bytes: usize, _stream: swamp_gpu::CudaStream) -> Option<*mut u8> {
        Some(std::ptr::null_mut())
    }

    pub fn upload_kv_async(&mut self, h_k: &[f32], h_v: &[f32], pos: usize) -> bool {
        let ctx = match self.ctx.as_mut() { Some(c) => c, None => return false };
        let kv = match ctx.kv_cache.as_ref() { Some(k) => k, None => return false };
        let staging = match ctx.staging.as_ref() { Some(s) => s, None => return false };
        let wpos = pos % self.window_size;
        let elem_size = 2u64;
        let kv_bytes = self.n_kv_heads as u64 * self.head_dim as u64 * elem_size;
        let offset = (wpos as u64) * kv_bytes;

        staging.buf.copy_from_host(ptr_as_bytes(h_k));
        ctx.copy_between(staging.buf.buffer, kv.k_buf.buffer, 0, offset, kv_bytes);

        staging.buf.copy_from_host(ptr_as_bytes(h_v));
        ctx.copy_between(staging.buf.buffer, kv.v_buf.buffer, 0, offset, kv_bytes);
        true
    }

    pub fn execute_attention_async(
        &mut self,
        h_q: &[f32],
        h_out: &mut [f32],
        _pos: usize,
        seq_len: usize,
    ) -> bool {
        let device = match self.device.as_ref() { Some(d) => d, None => return false };
        if !device.enabled { return false; }

        let (d_q, d_k, d_v, d_scores, d_attn, staging_buf) = {
            let ctx = match self.ctx.as_mut() { Some(c) => c, None => return false };
            let scratch = match ctx.scratch.as_ref() { Some(s) => s, None => return false };
            let kv = match ctx.kv_cache.as_ref() { Some(k) => k, None => return false };
            let staging = match ctx.staging.as_ref() { Some(s) => s, None => return false };

            let q_bytes = (self.num_heads * self.head_dim * 4) as u64;
            staging.buf.copy_from_host(ptr_as_bytes(h_q));
            ctx.copy_between(staging.buf.buffer, scratch.d_q.buffer, 0, 0, q_bytes);

            let attn_len = seq_len.min(self.window_size) as u32;
            if attn_len == 0 { return false; }

            let mut cg = device.compute_graph.lock().unwrap();
            cg.add_node(swamp_gpu::ComputeNodeOp::Attention {
                d_q: scratch.d_q.buffer, d_k: kv.k_buf.buffer, d_v: kv.v_buf.buffer,
                d_scores: scratch.d_scores.buffer, d_out: scratch.d_attn.buffer,
                n_heads: self.num_heads as u32, n_kv_heads: self.n_kv_heads as u32,
                seq_len: attn_len, head_dim: self.head_dim as u32,
                kv_stride: self.window_size as u32,
            }, vec![]);
            ctx.node_count += 1;
            drop(cg);

            (scratch.d_q.buffer, kv.k_buf.buffer, kv.v_buf.buffer,
             scratch.d_scores.buffer, scratch.d_attn.buffer, staging.buf.buffer)
        };

        {
            let ctx = match self.ctx.as_mut() { Some(c) => c, None => return false };
            ctx.submit_and_wait();
        }

        let out_bytes = (self.num_heads * self.head_dim * 4) as u64;
        if let Some(ctx) = self.ctx.as_mut() {
            if let Some(staging) = ctx.staging.as_ref() {
                ctx.copy_between(d_attn, staging.buf.buffer, 0, 0, out_bytes);
                staging.buf.copy_to_host(ptr_as_bytes_mut(h_out));
            }
        }
        true
    }

    pub fn sync(&self) -> bool {
        match self.ctx.as_ref() { Some(c) => {
            let device = match self.device.as_ref() { Some(d) => d, None => return false };
            device.compute_graph.lock().unwrap().wait().is_ok()
        }, None => false }
    }

    pub fn upload_x(&self, _h_x: &[f32]) -> bool { false }
    pub fn download_x(&self, _h_x: &mut [f32]) -> bool { false }
    pub fn create_layer_graph(&mut self, _l: usize, _embed_dim: usize, _ffn_dim: usize) -> bool { self.is_operational() }
    pub fn execute_layer_fused(&mut self, _l: usize, _pos: usize) -> bool { false }

    pub fn gemv_qkv_async(
        &mut self, layer_idx: usize,
        _d_w_q: *const u8, _d_w_k: *const u8, _d_w_v: *const u8,
        h_x: &[f32], h_q: &mut [f32], h_k: &mut [f32], h_v: &mut [f32],
        n_rows_q: i32, n_rows_k: i32, n_rows_v: i32, n_blocks: i32,
    ) -> bool {
        let (w_q, w_k, w_v, d_x, d_q, d_k, d_v) = match self.prepare_qkv_bufs(layer_idx) {
            Some(t) => t, None => return false,
        };
        let x_bytes = (n_blocks as usize) * 256 * 4;
        let x_src = &h_x[..(n_blocks as usize * 256).min(h_x.len())];
        let staging = match copy_host(self, x_src) { Some(s) => s, None => return false };
        self.staging_copy_to_device(staging, d_x, x_bytes as u64);
        {
            let device = match self.device.as_ref() { Some(d) => d, None => return false };
            let mut cg = device.compute_graph.lock().unwrap();
            let n0 = cg.add_node(swamp_gpu::ComputeNodeOp::GEMVQ4K {
                d_w: w_q, d_x, d_out: d_q, n_rows: n_rows_q as u32, n_blocks: n_blocks as u32,
            }, vec![]);
            let n1 = cg.add_node(swamp_gpu::ComputeNodeOp::GEMVQ4K {
                d_w: w_k, d_x, d_out: d_k, n_rows: n_rows_k as u32, n_blocks: n_blocks as u32,
            }, vec![n0]);
            cg.add_node(swamp_gpu::ComputeNodeOp::GEMVQ4K {
                d_w: w_v, d_x, d_out: d_v, n_rows: n_rows_v as u32, n_blocks: n_blocks as u32,
            }, vec![n1]);
        }
        submit_and_wait_ctx(self);
        readback_qkv(self, h_q, h_k, h_v, d_q, d_k, d_v, n_rows_q as usize, n_rows_k as usize, n_rows_v as usize)
    }

    pub fn gemv_gate_up_async(
        &mut self, layer_idx: usize,
        _d_w_gate: *const u8, _d_w_up: *const u8,
        h_x: &[f32], h_gate: &mut [f32], h_up: &mut [f32],
        n_rows: i32, n_blocks: i32,
    ) -> bool {
        let (w_g, w_u, d_x, d_g, d_u) = match self.prepare_gate_up_bufs(layer_idx) {
            Some(t) => t, None => return false,
        };
        let x_bytes = (n_blocks as usize) * 256 * 4;
        let x_src = &h_x[..(n_blocks as usize * 256).min(h_x.len())];
        let out_bytes = (n_rows as usize) * 4;
        let staging = match copy_host(self, x_src) { Some(s) => s, None => return false };
        self.staging_copy_to_device(staging, d_x, x_bytes as u64);
        {
            let device = match self.device.as_ref() { Some(d) => d, None => return false };
            let mut cg = device.compute_graph.lock().unwrap();
            let n0 = cg.add_node(swamp_gpu::ComputeNodeOp::GEMVQ4K {
                d_w: w_g, d_x, d_out: d_g, n_rows: n_rows as u32, n_blocks: n_blocks as u32,
            }, vec![]);
            cg.add_node(swamp_gpu::ComputeNodeOp::GEMVQ4K {
                d_w: w_u, d_x, d_out: d_u, n_rows: n_rows as u32, n_blocks: n_blocks as u32,
            }, vec![n0]);
        }
        submit_and_wait_ctx(self);
        let stg = match staging_buf(self) { Some(b) => b, None => return false };
        copy_from_gpu(self, d_g, stg, out_bytes as u64);
        readback_f32(self, stg, h_gate, out_bytes / 4);
        copy_from_gpu(self, d_u, stg, out_bytes as u64);
        readback_f32(self, stg, h_up, out_bytes / 4);
        true
    }

    fn prepare_qkv_bufs(&mut self, l: usize) -> Option<(vk::Buffer, vk::Buffer, vk::Buffer, vk::Buffer, vk::Buffer, vk::Buffer, vk::Buffer)> {
        let ctx = self.ctx.as_mut()?;
        let scratch = ctx.scratch.as_ref()?;
        let wq = ctx.weights.q.get(l)?.buffer;
        let wk = ctx.weights.k.get(l)?.buffer;
        let wv = ctx.weights.v.get(l)?.buffer;
        if wq == vk::Buffer::null() || wk == vk::Buffer::null() || wv == vk::Buffer::null() { return None; }
        Some((wq, wk, wv, scratch.d_x.buffer, scratch.d_q.buffer, scratch.d_k.buffer, scratch.d_v.buffer))
    }

    fn prepare_gate_up_bufs(&mut self, l: usize) -> Option<(vk::Buffer, vk::Buffer, vk::Buffer, vk::Buffer, vk::Buffer)> {
        let ctx = self.ctx.as_mut()?;
        let scratch = ctx.scratch.as_ref()?;
        let wg = ctx.weights.gate.get(l)?.buffer;
        let wu = ctx.weights.up.get(l)?.buffer;
        if wg == vk::Buffer::null() || wu == vk::Buffer::null() { return None; }
        Some((wg, wu, scratch.d_x.buffer, scratch.d_gate.buffer, scratch.d_up.buffer))
    }

    pub fn execute_gemv_async(
        &mut self, layer_idx: usize, _d_w: *const u8,
        h_x: &[f32], h_out: &mut [f32], n_rows: i32, n_blocks: i32,
    ) -> bool {
        let weight_buf = match self.get_weight_buf(layer_idx, "single") { Some(b) => b, None => return false };
        let x_bytes = (n_blocks as usize) * 256 * 4;
        let out_bytes = (n_rows as usize) * 4;
        let x_src = &h_x[..(n_blocks as usize * 256).min(h_x.len())];
        self.gemv_one(weight_buf, x_src, h_out, x_bytes, out_bytes, n_rows as u32, n_blocks as u32)
    }

    fn get_weight_buf(&self, layer: usize, op: &str) -> Option<vk::Buffer> {
        let ctx = self.ctx.as_ref()?;
        let candidates: &[&Vec<swamp_gpu::GpuBuf>] = match op {
            "o" => &[&ctx.weights.o],
            "down" => &[&ctx.weights.down],
            "single" => &[&ctx.weights.o, &ctx.weights.down, &ctx.weights.gate],
            _ => return None,
        };
        for arr in candidates {
            if let Some(w) = arr.get(layer) {
                if w.buffer != vk::Buffer::null() { return Some(w.buffer); }
            }
        }
        None
    }

    fn gemv_one(&mut self, w: vk::Buffer, h_x: &[f32], h_out: &mut [f32], x_bytes: usize, out_bytes: usize, n_rows: u32, n_blocks: u32) -> bool {
        let (d_x, d_out) = {
            let ctx = match self.ctx.as_mut() { Some(c) => c, None => return false };
            let scratch = match ctx.scratch.as_ref() { Some(s) => s, None => return false };
            (scratch.d_x.buffer, scratch.d_down.buffer)
        };
        let staging = match copy_host(self, h_x) { Some(s) => s, None => return false };
        self.staging_copy_to_device(staging, d_x, x_bytes as u64);
        {
            let device = match self.device.as_ref() { Some(d) => d, None => return false };
            let mut cg = device.compute_graph.lock().unwrap();
            cg.add_node(swamp_gpu::ComputeNodeOp::GEMVQ4K { d_w: w, d_x, d_out, n_rows, n_blocks }, vec![]);
        }
        submit_and_wait_ctx(self);
        let stg = match staging_buf(self) { Some(b) => b, None => return false };
        copy_from_gpu(self, d_out, stg, out_bytes as u64);
        readback_f32(self, stg, h_out, out_bytes / 4);
        true
    }
}

/// Copy host data to staging buffer and return the staging buffer handle
fn staging_buf(s: &mut PerLayerGpuState) -> Option<vk::Buffer> {
    let ctx = s.ctx.as_mut()?;
    Some(ctx.staging.as_ref()?.buf.buffer)
}

fn copy_host(s: &mut PerLayerGpuState, src: &[f32]) -> Option<vk::Buffer> {
    let buf = staging_buf(s)?;
    let ctx = s.ctx.as_mut()?;
    let staging = ctx.staging.as_ref()?;
    staging.buf.copy_from_host(ptr_as_bytes(src));
    Some(buf)
}

fn copy_to_gpu(s: &mut PerLayerGpuState, staging: vk::Buffer, dst: vk::Buffer, size: u64) -> bool {
    let ctx = match s.ctx.as_mut() { Some(c) => c, None => return false };
    ctx.copy_between(staging, dst, 0, 0, size)
}

fn copy_from_gpu(s: &mut PerLayerGpuState, src: vk::Buffer, staging: vk::Buffer, size: u64) -> bool {
    let ctx = match s.ctx.as_mut() { Some(c) => c, None => return false };
    ctx.copy_between(src, staging, 0, 0, size)
}

fn readback_f32(s: &mut PerLayerGpuState, staging: vk::Buffer, dst: &mut [f32], count: usize) -> bool {
    let ctx = match s.ctx.as_mut() { Some(c) => c, None => return false };
    let stg = match ctx.staging.as_ref() { Some(st) => st, None => return false };
    let len = count.min(dst.len());
    stg.buf.copy_to_host(ptr_as_bytes_mut(&mut dst[..len]));
    true
}

fn submit_and_wait_ctx(s: &mut PerLayerGpuState) -> bool {
    let ctx = match s.ctx.as_mut() { Some(c) => c, None => return false };
    ctx.submit_and_wait()
}

fn readback_qkv(s: &mut PerLayerGpuState, h_q: &mut [f32], h_k: &mut [f32], h_v: &mut [f32], d_q: vk::Buffer, d_k: vk::Buffer, d_v: vk::Buffer, nq: usize, nk: usize, nv: usize) -> bool {
    let stg = match staging_buf(s) { Some(b) => b, None => return false };
    let qb = nq * 4; let kb = nk * 4; let vb = nv * 4;
    copy_from_gpu(s, d_q, stg, qb as u64);
    readback_f32(s, stg, h_q, nq);
    copy_from_gpu(s, d_k, stg, kb as u64);
    readback_f32(s, stg, h_k, nk);
    copy_from_gpu(s, d_v, stg, vb as u64);
    readback_f32(s, stg, h_v, nv);
    true
}

impl PerLayerGpuState {
    /// Create PerLayerGpuState reusing an existing GpuDevice (shared with DataPlane)
    pub fn with_device(
        device: &Arc<swamp_gpu::GpuDevice>,
        window_size: usize, num_heads: usize, n_kv_heads: usize, head_dim: usize,
        num_layers: usize, embed_dim: usize, ffn_dim: usize,
    ) -> Option<Self> {
        let ctx_opt = swamp_gpu::GpuComputeContext::new(
            device, embed_dim, ffn_dim, num_heads, n_kv_heads, head_dim, window_size, num_layers,
        );
        let enabled = ctx_opt.is_some();
        Some(Self {
            ctx: ctx_opt, device: Some(device.clone()), enabled,
            num_layers, window_size, num_heads, n_kv_heads, head_dim, embed_dim, ffn_dim,
        })
    }

    /// Upload all weight tensors from model RingView data.
    /// Call after new() and before any GEMV operations.
    pub fn upload_all_weights(&mut self, model: &crate::model::Model) -> bool {
        let ctx = match self.ctx.as_mut() { Some(c) => c, None => return false };
        for l in 0..self.num_layers.min(model.layer_rings.len()) {
            let r = &model.layer_rings[l];
            let ok = ctx.upload_layer_weights(
                l,
                r.q_slice(), r.k_slice(), r.v_slice(),
                r.o_slice(), r.gate_slice(), r.up_slice(), r.down_slice(),
                &[], &[], // attn_norm and ffn_norm are uploaded separately
            );
            if !ok { return false; }
        }
        true
    }

    fn staging_copy_to_device(&mut self, staging: vk::Buffer, dst: vk::Buffer, size: u64) -> bool {
        let ctx = match self.ctx.as_mut() { Some(c) => c, None => return false };
        ctx.copy_between(staging, dst, 0, 0, size)
    }

    pub fn upload_norm_weights(&mut self, h_attn_norms: &[Vec<f32>], h_ffn_norms: &[Vec<f32>]) {
        let ctx = match self.ctx.as_mut() { Some(c) => c, None => return };
        for l in 0..self.num_layers.min(h_attn_norms.len()) {
            if l >= ctx.weights.attn_norm.len() { break; }
            ctx.upload_norm_weight(l, true, ptr_as_bytes(&h_attn_norms[l]));
        }
        for l in 0..self.num_layers.min(h_ffn_norms.len()) {
            if l >= ctx.weights.ffn_norm.len() { break; }
            ctx.upload_norm_weight(l, false, ptr_as_bytes(&h_ffn_norms[l]));
        }
    }
}

impl Drop for PerLayerGpuState {
    fn drop(&mut self) {}
}

fn ptr_as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<T>()) }
}

fn ptr_as_bytes_mut<T>(v: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * std::mem::size_of::<T>()) }
}

// ---------------------------------------------------------------------------
// WorkToken: lightweight RAII guard for tracking pipeline depth
// ---------------------------------------------------------------------------

pub struct PipelineToken {
    counter: &'static AtomicU32,
}

impl PipelineToken {
    pub fn acquire(counter: &'static AtomicU32, max_depth: u32) -> Option<Self> {
        loop {
            let current = counter.load(Ordering::Relaxed);
            if current >= max_depth {
                return None;
            }
            if counter.compare_exchange_weak(current, current + 1, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                return Some(PipelineToken { counter });
            }
        }
    }
}

impl Drop for PipelineToken {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_token() {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let t1 = PipelineToken::acquire(&COUNTER, 2).unwrap();
        assert_eq!(COUNTER.load(Ordering::Relaxed), 1);
        let t2 = PipelineToken::acquire(&COUNTER, 2).unwrap();
        assert_eq!(COUNTER.load(Ordering::Relaxed), 2);
        let t3 = PipelineToken::acquire(&COUNTER, 2);
        assert!(t3.is_none());
        drop(t2);
        assert_eq!(COUNTER.load(Ordering::Relaxed), 1);
        drop(t1);
        assert_eq!(COUNTER.load(Ordering::Relaxed), 0);
    }
}
