use std::sync::{OnceLock, Mutex};
use std::cell::RefCell;
use std::collections::HashMap;
use rayon::prelude::*;
use swamp_kernels::simd::{dot_product_raw, weighted_sum_raw};

thread_local! {
    static SCORES_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

const ROPE_THETA: f32 = 10000.0;

/// LUT de RoPE: mapeia head_dim -> Vec<[cos, sin]> (flat: [pos * half_dim + k][cos, sin])
static ROPE_LUTS: OnceLock<Mutex<HashMap<usize, Vec<[f32; 2]>>>> = OnceLock::new();

fn rope_luts() -> &'static Mutex<HashMap<usize, Vec<[f32; 2]>>> {
    ROPE_LUTS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn init_rope_lut(head_dim: usize, max_seq_len: usize) {
    let half_dim = head_dim / 2;
    let mut luts = rope_luts().lock().unwrap();
    if luts.contains_key(&head_dim) {
        return;
    }

    let mut lut = vec![[0.0f32; 2]; max_seq_len * half_dim];
    for m in 0..max_seq_len {
        for k in 0..half_dim {
            let freq = 1.0 / (ROPE_THETA.powf((2 * k) as f32 / head_dim as f32));
            let angle = m as f32 * freq;
            let idx = m * half_dim + k;
            lut[idx][0] = angle.cos();
            lut[idx][1] = angle.sin();
        }
    }
    luts.insert(head_dim, lut);
}

/// Aplica RoPE in-place para um token
#[inline(always)]
pub fn apply_rope_ufc(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_context: usize,
) {
    let half_dim = head_dim / 2;
    let luts = rope_luts().lock().unwrap();
    let lut = luts.get(&head_dim).expect("RoPE LUT not initialized for this head_dim");
    let pos = pos.min(max_context - 1);
    let lut_base = pos * half_dim;

    for h in 0..num_q_heads {
        let base = h * head_dim;
        apply_rotation_pair(&mut q[base..base + head_dim], &lut[lut_base..lut_base + half_dim], half_dim);
    }

    for h in 0..num_kv_heads {
        let base = h * head_dim;
        apply_rotation_pair(&mut k[base..base + head_dim], &lut[lut_base..lut_base + half_dim], half_dim);
    }
}

#[inline(always)]
fn apply_rotation_pair(vec: &mut [f32], cos_sin: &[[f32; 2]], half_dim: usize) {
    for k in 0..half_dim {
        let i = k * 2;
        let x = vec[i];
        let y = vec[i + 1];
        let cos = cos_sin[k][0];
        let sin = cos_sin[k][1];

        vec[i] = x * cos - y * sin;
        vec[i + 1] = x * sin + y * cos;
    }
}

/// Sparse attention: window + global tokens.
/// At seq_len=1M, attends to last `window` tokens + every `global_stride`-th older token.
/// This reduces O(n²) to O(window × n_heads) per token.
pub fn attention_sparse(
    out: &mut [f32],
    q: &[f32],
    kv_cache: &mut crate::cache::PagedKVCache,
    layer_idx: usize,
    seq_len: usize,
    q_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    global_stride: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_rep = n_heads / n_kv_heads;
    let valid_end = q_pos.min(seq_len - 1);

    // Build sparse position list: last `window` + every `global_stride`-th older
    let mut positions: Vec<usize> = Vec::with_capacity(window + 128);
    let window_start = if valid_end > window { valid_end - window + 1 } else { 0 };
    // Global tokens: sample evenly from [0, window_start)
    let global_count = window_start / global_stride;
    for gi in 0..global_count.min(128) {
        positions.push(global_stride * gi);
    }
    // Window tokens
    for t in window_start..=valid_end {
        positions.push(t);
    }

    let end_page = kv_cache.page_id(valid_end);
    kv_cache.ensure_pages_hot_f32(0, end_page);

    out.par_chunks_mut(head_dim)
       .enumerate()
       .for_each(|(h, out_head)| {
            let kv_h = h / n_rep;
            let q_ptr = unsafe { q.as_ptr().add(h * head_dim) };
            let k = positions.len();

            SCORES_BUF.with(|buf| {
                let mut buf_ref = buf.borrow_mut();
                if buf_ref.len() < k {
                    buf_ref.resize(k, 0.0);
                }
                let scores = &mut buf_ref[..k];

                // Score loop (sparse positions)
                let mut max_score = f32::NEG_INFINITY;
                for (idx, &t) in positions.iter().enumerate() {
                    let pid = kv_cache.page_id(t);
                    let page_start = kv_cache.page_start_for(t);
                    let k_page_base = kv_cache.k_page_ptr(layer_idx, kv_h, pid);
                    let k_ptr = unsafe { k_page_base.add((t - page_start) * head_dim) };
                    let score = unsafe { dot_product_raw(q_ptr, k_ptr, head_dim) * scale };
                    scores[idx] = score;
                    if score > max_score { max_score = score; }
                }

                // Softmax over sparse set
                let mut sum_exp = 0.0f32;
                for s in scores.iter_mut() {
                    let e = (*s - max_score).exp();
                    *s = e;
                    sum_exp += e;
                }
                let inv_sum = 1.0 / sum_exp;
                for s in scores.iter_mut() { *s *= inv_sum; }

                // Weighted sum from sparse positions
                unsafe {
                    let out_ptr = out_head.as_mut_ptr();
                    std::ptr::write_bytes(out_ptr, 0, head_dim);
                    for (idx, &t) in positions.iter().enumerate() {
                        let pid = kv_cache.page_id(t);
                        let page_start = kv_cache.page_start_for(t);
                        let v_page_base = kv_cache.v_page_ptr(layer_idx, kv_h, pid);
                        let v_ptr = v_page_base.add((t - page_start) * head_dim);
                        weighted_sum_raw(out_ptr, v_ptr, scores[idx], head_dim);
                    }
                }
            });
        });
}

pub fn attention(
    out: &mut [f32],
    q: &[f32],
    kv_cache: &mut crate::cache::PagedKVCache,
    layer_idx: usize,
    seq_len: usize,
    q_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_rep = n_heads / n_kv_heads;
    let valid_end = q_pos.min(seq_len - 1);

    let end_page = kv_cache.page_id(valid_end);
    kv_cache.ensure_pages_hot_f32(0, end_page);

    out.par_chunks_mut(head_dim)
       .enumerate()
       .for_each(|(h, out_head)| {
            let kv_h = h / n_rep;
            let q_ptr = unsafe { q.as_ptr().add(h * head_dim) };

            SCORES_BUF.with(|buf| {
                let mut scores = buf.borrow_mut();
                if scores.len() < seq_len {
                    scores.resize(seq_len, 0.0);
                }
                let scores = &mut scores[..seq_len];

                // === SCORE LOOP (page by page, zone-aware) ===
                let mut max_score = f32::NEG_INFINITY;
                let mut t = 0;
                while t <= valid_end {
                    let pid = kv_cache.page_id(t);
                    let page_start = kv_cache.page_start_for(t);
                    let bs = kv_cache.block_size_for(t);
                    let page_end = (page_start + bs - 1).min(valid_end);

                    let k_page_base = kv_cache.k_page_ptr(layer_idx, kv_h, pid);

                    unsafe {
                        for tt in page_start..=page_end {
                            #[cfg(target_arch = "x86_64")]
                            if tt + 4 <= page_end {
                                std::arch::x86_64::_mm_prefetch(
                                    k_page_base.add((tt + 4 - page_start) * head_dim) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T0,
                                );
                            }
                            let k_ptr = k_page_base.add((tt - page_start) * head_dim);
                            let score = dot_product_raw(q_ptr, k_ptr, head_dim) * scale;
                            scores[tt] = score;
                            if score > max_score {
                                max_score = score;
                            }
                        }
                    }
                    t = page_end + 1;
                }

                // Mascara posicoes futuras (causal)
                for t in (valid_end + 1)..seq_len {
                    scores[t] = f32::NEG_INFINITY;
                }

                // Softmax
                let mut sum_exp = 0.0f32;
                for t in 0..seq_len {
                    let exp_score = (scores[t] - max_score).exp();
                    scores[t] = exp_score;
                    sum_exp += exp_score;
                }
                let inv_sum = 1.0 / sum_exp;
                for t in 0..seq_len {
                    scores[t] *= inv_sum;
                }

                // === WEIGHTED SUM LOOP (page by page, zone-aware) ===
                unsafe {
                    let out_ptr = out_head.as_mut_ptr();
                    std::ptr::write_bytes(out_ptr, 0, head_dim);
                    let mut t = 0;
                    while t <= valid_end {
                        let pid = kv_cache.page_id(t);
                        let page_start = kv_cache.page_start_for(t);
                        let bs = kv_cache.block_size_for(t);
                        let page_end = (page_start + bs - 1).min(valid_end);

                        let v_page_base = kv_cache.v_page_ptr(layer_idx, kv_h, pid);

                        for tt in page_start..=page_end {
                            #[cfg(target_arch = "x86_64")]
                            if tt + 4 <= page_end {
                                std::arch::x86_64::_mm_prefetch(
                                    v_page_base.add((tt + 4 - page_start) * head_dim) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T0,
                                );
                            }
                            let v_ptr = v_page_base.add((tt - page_start) * head_dim);
                            weighted_sum_raw(out_ptr, v_ptr, scores[tt], head_dim);
                        }
                        t = page_end + 1;
                    }
                }
            });
        });
}

/// Full attention that also records per-position softmax weights for
/// entropy-based eviction. `importance` is accumulated across all heads:
/// positions with high cumulative weight have low entropy (heavy hitters).
pub fn attention_tracked(
    out: &mut [f32],
    q: &[f32],
    kv_cache: &mut crate::cache::PagedKVCache,
    layer_idx: usize,
    seq_len: usize,
    q_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    importance: &mut [f64],
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_rep = n_heads / n_kv_heads;
    let valid_end = q_pos.min(seq_len - 1);

    let end_page = kv_cache.page_id(valid_end);
    kv_cache.ensure_pages_hot_f32(0, end_page);

    // Use an independent Vec inside Mutex to avoid &mut [f64] Sync issues in rayón
    let acc = std::sync::Mutex::new(importance.to_vec());

    out.par_chunks_mut(head_dim)
       .enumerate()
       .for_each(|(h, out_head)| {
            let kv_h = h / n_rep;
            let q_ptr = unsafe { q.as_ptr().add(h * head_dim) };

            SCORES_BUF.with(|buf| {
                let mut scores = buf.borrow_mut();
                if scores.len() < seq_len {
                    scores.resize(seq_len, 0.0);
                }
                let scores = &mut scores[..seq_len];

                // === SCORE LOOP (same as attention) ===
                let mut max_score = f32::NEG_INFINITY;
                let mut t = 0;
                while t <= valid_end {
                    let pid = kv_cache.page_id(t);
                    let page_start = kv_cache.page_start_for(t);
                    let bs = kv_cache.block_size_for(t);
                    let page_end = (page_start + bs - 1).min(valid_end);
                    let k_page_base = kv_cache.k_page_ptr(layer_idx, kv_h, pid);
                    unsafe {
                        for tt in page_start..=page_end {
                            #[cfg(target_arch = "x86_64")]
                            if tt + 4 <= page_end {
                                std::arch::x86_64::_mm_prefetch(
                                    k_page_base.add((tt + 4 - page_start) * head_dim) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T0,
                                );
                            }
                            let k_ptr = k_page_base.add((tt - page_start) * head_dim);
                            let score = dot_product_raw(q_ptr, k_ptr, head_dim) * scale;
                            scores[tt] = score;
                            if score > max_score { max_score = score; }
                        }
                    }
                    t = page_end + 1;
                }

                // Mask future positions
                for t in (valid_end + 1)..seq_len {
                    scores[t] = f32::NEG_INFINITY;
                }

                // Softmax
                let mut sum_exp = 0.0f32;
                for t in 0..seq_len {
                    let exp_score = (scores[t] - max_score).exp();
                    scores[t] = exp_score;
                    sum_exp += exp_score;
                }
                let inv_sum = 1.0 / sum_exp;
                for t in 0..seq_len {
                    scores[t] *= inv_sum;
                }

                // Accumulate importance across heads
                {
                    let mut guard = acc.lock().unwrap();
                    for t in 0..seq_len {
                        guard[t] += scores[t] as f64;
                    }
                }

                // === WEIGHTED SUM (same as attention) ===
                unsafe {
                    let out_ptr = out_head.as_mut_ptr();
                    std::ptr::write_bytes(out_ptr, 0, head_dim);
                    let mut t = 0;
                    while t <= valid_end {
                        let pid = kv_cache.page_id(t);
                        let page_start = kv_cache.page_start_for(t);
                        let bs = kv_cache.block_size_for(t);
                        let page_end = (page_start + bs - 1).min(valid_end);
                        let v_page_base = kv_cache.v_page_ptr(layer_idx, kv_h, pid);
                        for tt in page_start..=page_end {
                            #[cfg(target_arch = "x86_64")]
                            if tt + 4 <= page_end {
                                std::arch::x86_64::_mm_prefetch(
                                    v_page_base.add((tt + 4 - page_start) * head_dim) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T0,
                                );
                            }
                            let v_ptr = v_page_base.add((tt - page_start) * head_dim);
                            weighted_sum_raw(out_ptr, v_ptr, scores[tt], head_dim);
                        }
                        t = page_end + 1;
                    }
                }
            });
        });

    // Copy accumulated weights back to caller
    let final_acc = acc.into_inner().unwrap();
    importance.copy_from_slice(&final_acc);
}

pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0.0f32;
    for i in 0..n {
        ss += x[i] * x[i];
    }
    ss /= n as f32;
    ss += eps;
    let inv_rms = 1.0 / ss.sqrt();

    for i in 0..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

pub fn silu(x: &mut [f32]) {
    for i in 0..x.len() {
        let val = x[i];
        x[i] = val / (1.0 + (-val).exp());
    }
}

pub fn mul_in_place(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] *= b[i];
    }
}

pub fn add_in_place(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

/// Fugu hierarchical sparse attention:
///   - sliding window (last `window` tokens)
///   - sentinel tokens (every `sentinel_stride`-th older token)
///   - DSPark-selected cold blocks (positions from LSH match)
///
/// Combines all three into a single sparse position set, then computes
/// attention (score + softmax + weighted sum) via the FP32 dequant path.
/// Best for seq_len > sparse_attention_threshold where full O(n²) is infeasible.
pub fn attention_fugu(
    out: &mut [f32],
    q: &[f32],
    kv_cache: &mut crate::cache::PagedKVCache,
    layer_idx: usize,
    seq_len: usize,
    q_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    sentinel_stride: usize,
    dspark_positions: &[usize],
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_rep = n_heads / n_kv_heads;
    let valid_end = q_pos.min(seq_len - 1);

    // 1. Build position set
    let window_start = if valid_end > window { valid_end - window + 1 } else { 0 };
    let max_extra = window + 128 + dspark_positions.len();
    let mut positions: Vec<usize> = Vec::with_capacity(max_extra);

    // Sentinel tokens (every sentinel_stride-th from early context)
    let sentinel_count = window_start / sentinel_stride;
    for gi in 0..sentinel_count.min(128) {
        positions.push(sentinel_stride * gi);
    }

    // Window tokens (most recent positions)
    for t in window_start..=valid_end {
        positions.push(t);
    }

    // DSPark-selected cold block positions (dedup against existing)
    for &dp in dspark_positions {
        if dp <= valid_end && !positions.contains(&dp) {
            positions.push(dp);
        }
    }

    // 2. Ensure all needed pages are hot + dequantized
    let end_page = kv_cache.page_id(valid_end);
    kv_cache.ensure_pages_hot_f32(0, end_page);

    // 3. Parallel attention over sparse positions
    let k = positions.len();
    out.par_chunks_mut(head_dim)
       .enumerate()
       .for_each(|(h, out_head)| {
            let kv_h = h / n_rep;
            let q_ptr = unsafe { q.as_ptr().add(h * head_dim) };

            SCORES_BUF.with(|buf| {
                let mut buf_ref = buf.borrow_mut();
                if buf_ref.len() < k {
                    buf_ref.resize(k, 0.0);
                }
                let scores = &mut buf_ref[..k];

                // Score loop
                let mut max_score = f32::NEG_INFINITY;
                for (idx, &t) in positions.iter().enumerate() {
                    let pid = kv_cache.page_id(t);
                    let page_start = kv_cache.page_start_for(t);
                    let k_page_base = kv_cache.k_page_ptr(layer_idx, kv_h, pid);
                    let k_ptr = unsafe { k_page_base.add((t - page_start) * head_dim) };
                    let score = unsafe { dot_product_raw(q_ptr, k_ptr, head_dim) * scale };
                    scores[idx] = score;
                    if score > max_score { max_score = score; }
                }

                // Softmax over sparse set
                let mut sum_exp = 0.0f32;
                for s in scores.iter_mut() {
                    let e = (*s - max_score).exp();
                    *s = e;
                    sum_exp += e;
                }
                let inv_sum = 1.0 / sum_exp;
                for s in scores.iter_mut() { *s *= inv_sum; }

                // Weighted sum from sparse positions
                unsafe {
                    let out_ptr = out_head.as_mut_ptr();
                    std::ptr::write_bytes(out_ptr, 0, head_dim);
                    for (idx, &t) in positions.iter().enumerate() {
                        let pid = kv_cache.page_id(t);
                        let page_start = kv_cache.page_start_for(t);
                        let v_page_base = kv_cache.v_page_ptr(layer_idx, kv_h, pid);
                        let v_ptr = v_page_base.add((t - page_start) * head_dim);
                        weighted_sum_raw(out_ptr, v_ptr, scores[idx], head_dim);
                    }
                }
            });
        });
}

/// GPU-accelerated attention (falls back to CPU if GPU unavailable)
#[cfg(feature = "gpu")]
pub fn gpu_attention_forward(
    out: &mut [f32],
    q: &[f32],
    kv_cache: &mut crate::cache::PagedKVCache,
    layer_idx: usize,
    seq_len: usize,
    _q_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    // Allocate contiguous K and V buffers
    let kv_size = n_kv_heads * seq_len * head_dim;
    let mut k_buf = vec![0.0f32; kv_size];
    let mut v_buf = vec![0.0f32; kv_size];

    tracing::debug!("GPU attention: seq_len={} heads={}/{} head_dim={}",
        seq_len, n_heads, n_kv_heads, head_dim);

    // Copy K and V from page cache into contiguous buffers
    for kv_h in 0..n_kv_heads {
        for t in 0..seq_len {
            let pid = kv_cache.page_id(t);
            let src_k = kv_cache.k_page_ptr(layer_idx, kv_h, pid);
            let src_v = kv_cache.v_page_ptr(layer_idx, kv_h, pid);
            let slot = kv_cache.slot_in_page(t);
            unsafe {
                let dst_off = (kv_h * seq_len + t) * head_dim;
                std::ptr::copy_nonoverlapping(
                    src_k.add(slot * head_dim),
                    k_buf.as_mut_ptr().add(dst_off),
                    head_dim,
                );
                std::ptr::copy_nonoverlapping(
                    src_v.add(slot * head_dim),
                    v_buf.as_mut_ptr().add(dst_off),
                    head_dim,
                );
            }
        }
    }

    // Call GPU attention
    if let Err(e) = swamp_gpu::gpu_attention_forward(q, &k_buf, &v_buf, out, n_heads, n_kv_heads, seq_len, head_dim) {
        tracing::warn!("GPU attention failed, falling back to CPU: {:?}", e);
        attention(out, q, kv_cache, layer_idx, seq_len, _q_pos, n_heads, n_kv_heads, head_dim);
    }
}

/// Sparse attention with 4-bit KV cache.
/// Reads K/V from q4 pages, dequantizes on-the-fly during score/weighted-sum.
/// Window + global strategy identical to attention_sparse.
pub fn attention_sparse_q4(
    out: &mut [f32],
    q: &[f32],
    kv_cache: &mut crate::cache::PagedKVCache,
    layer_idx: usize,
    seq_len: usize,
    q_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    global_stride: usize,
) {
    use std::arch::x86_64::*;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_rep = n_heads / n_kv_heads;
    let valid_end = q_pos.min(seq_len - 1);

    // Build sparse positions
    let mut positions: Vec<usize> = Vec::with_capacity(window + 128);
    let window_start = if valid_end > window { valid_end - window + 1 } else { 0 };
    let global_count = window_start / global_stride;
    for gi in 0..global_count.min(128) {
        positions.push(global_stride * gi);
    }
    for t in window_start..=valid_end { positions.push(t); }
    let k = positions.len();

    let blocks_per_head = (head_dim + 31) / 32;
    let blk_bytes = 20; // d(2)+dmin(2)+nibbles(16)

    out.par_chunks_mut(head_dim).enumerate().for_each(|(h, out_head)| {
        let kv_h = h / n_rep;
        let q_f32 = unsafe { std::slice::from_raw_parts(q.as_ptr().add(h * head_dim), head_dim) };
        let mask_nib = unsafe { _mm_set1_epi8(0x0F) };

        // Score loop over sparse positions (4-bit K, AVX-512 vectorized)
        let mut scores = vec![0.0f32; k];
        let mut max_score = f32::NEG_INFINITY;
        for (idx, &t) in positions.iter().enumerate() {
            let pid = kv_cache.page_id(t);
            let slot = kv_cache.slot_in_page(t);
            let mut dot_acc = unsafe { _mm512_setzero_ps() };

            for blk in 0..blocks_per_head {
                let q4_page = kv_cache.k_q4_page_ptr(layer_idx, kv_h, pid);
                let blk_off = (slot * blocks_per_head + blk) * blk_bytes;

                unsafe {
                    let base = q4_page.add(blk_off);
                    let d = half::f16::from_le_bytes([*base, *base.add(1)]).to_f32();
                    let dmin = half::f16::from_le_bytes([*base.add(2), *base.add(3)]).to_f32();
                    let nib = base.add(4);
                    let d_ps = _mm512_set1_ps(d);
                    let dmin_ps = _mm512_set1_ps(dmin);

                    // Load 16 nibble bytes, expand to 32 nibbles (low/high)
                    let nib16 = _mm_loadu_si128(nib as *const __m128i);
                    let nib_lo = _mm_and_si128(nib16, mask_nib);
                    let nib_hi = _mm_and_si128(_mm_srli_epi16(nib16, 4), mask_nib);

                    // Extend nibbles to float, dequantize, FMA dot with Q
                    let k_f_lo = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(nib_lo));
                    let k_f_hi = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(nib_hi));
                    let k_val_lo = _mm512_fmadd_ps(d_ps, k_f_lo, dmin_ps);
                    let k_val_hi = _mm512_fmadd_ps(d_ps, k_f_hi, dmin_ps);

                    let q_lo = _mm512_loadu_ps(q_f32.as_ptr().add(blk * 32));
                    let q_hi = _mm512_loadu_ps(q_f32.as_ptr().add(blk * 32 + 16));
                    dot_acc = _mm512_fmadd_ps(k_val_lo, q_lo, dot_acc);
                    dot_acc = _mm512_fmadd_ps(k_val_hi, q_hi, dot_acc);
                }
            }

            // Horizontal reduction once per position
            let mut tmp = [0.0f32; 16];
            unsafe { _mm512_storeu_ps(tmp.as_mut_ptr(), dot_acc); }
            let dot: f32 = tmp.iter().sum();
            let score = dot * scale;
            scores[idx] = score;
            if score > max_score { max_score = score; }
        }

        // Softmax over sparse positions
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut() { let e = (*s - max_score).exp(); *s = e; sum_exp += e; }
        let inv_sum = 1.0 / sum_exp;
        for s in scores.iter_mut() { *s *= inv_sum; }

        // Weighted sum from 4-bit V (AVX-512 vectorized)
        unsafe { std::ptr::write_bytes(out_head.as_mut_ptr(), 0, head_dim); }
        for (idx, &t) in positions.iter().enumerate() {
            let pid = kv_cache.page_id(t);
            let slot = kv_cache.slot_in_page(t);
            let w = scores[idx];
            if w < 1e-8 { continue; }
            let w_ps = unsafe { _mm512_set1_ps(w) };

            for blk in 0..blocks_per_head {
                let q4_page = kv_cache.v_q4_page_ptr(layer_idx, kv_h, pid);
                let blk_off = (slot * blocks_per_head + blk) * blk_bytes;
                unsafe {
                    let base = q4_page.add(blk_off);
                    let d = half::f16::from_le_bytes([*base, *base.add(1)]).to_f32();
                    let dmin = half::f16::from_le_bytes([*base.add(2), *base.add(3)]).to_f32();
                    let nib = base.add(4);
                    let d_ps = _mm512_set1_ps(d);
                    let dmin_ps = _mm512_set1_ps(dmin);

                    let nib16 = _mm_loadu_si128(nib as *const __m128i);
                    let nib_lo = _mm_and_si128(nib16, mask_nib);
                    let nib_hi = _mm_and_si128(_mm_srli_epi16(nib16, 4), mask_nib);

                    let v_f_lo = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(nib_lo));
                    let v_f_hi = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(nib_hi));
                    let v_val_lo = _mm512_fmadd_ps(d_ps, v_f_lo, dmin_ps);
                    let v_val_hi = _mm512_fmadd_ps(d_ps, v_f_hi, dmin_ps);

                    let out_lo = _mm512_loadu_ps(out_head.as_mut_ptr().add(blk * 32));
                    let out_hi = _mm512_loadu_ps(out_head.as_mut_ptr().add(blk * 32 + 16));
                    _mm512_storeu_ps(out_head.as_mut_ptr().add(blk * 32), _mm512_fmadd_ps(v_val_lo, w_ps, out_lo));
                    _mm512_storeu_ps(out_head.as_mut_ptr().add(blk * 32 + 16), _mm512_fmadd_ps(v_val_hi, w_ps, out_hi));
                }
            }
        }
    });
}
