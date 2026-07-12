use anyhow::{bail, Result};
use rayon::prelude::*;
use rand::Rng;
use std::sync::OnceLock;
use std::collections::HashMap;
use swamp_gguf::{TensorInfo, GgmlDType, GgufFile};
use swamp_kernels::fused_gemv_q4k::{fused_gemv_q4k, fused_gemv_q4k_batched};
use swamp_kernels::fused_gemv_q6k::fused_gemv_q6k;

const Q4K_BLOCK: usize = 256;
const Q4K_BYTES: usize = 144;

// =========================================================================
// Block-level precision sensitivity map (Idea #4)
// Maps tensor_name → (sensitive_row_indices, fp32_weight_data flat per row)
// =========================================================================
static SENSITIVITY_MAP: OnceLock<HashMap<String, (Vec<usize>, Vec<f32>)>> = OnceLock::new();

/// Set the global sensitivity map (called once at init after calibration).
pub fn set_sensitivity_map(map: HashMap<String, (Vec<usize>, Vec<f32>)>) {
    let _ = SENSITIVITY_MAP.set(map);
}

/// Calibrate row-level sensitivity: compare Q4_K output vs FP32 reference
/// and flag rows with relative error > threshold as sensitive.
/// Returns a map from tensor_name → (sensitive_row_indices, fp32_weight_data).
pub fn calibrate_row_sensitivity(gguf: &GgufFile, num_layers: usize, threshold: f32) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let mut map: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for l in 0..num_layers {
        for name in &["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
            let tname = format!("blk.{}.{}.weight", l, name);
            let tensor = match gguf.tensor_or_err(&tname) {
                Ok(t) => t,
                Err(_) => continue,
            };
            if tensor.dtype != GgmlDType::Q4_K { continue; }
            let nc = tensor.shape[0] as usize;
            let nr = tensor.shape[1] as usize;
            let raw = match gguf.tensor_raw_bytes(tensor) { Ok(r) => r, Err(_) => continue };
            let n_blocks = nc / Q4K_BLOCK;
            let row_bytes = n_blocks * Q4K_BYTES;
            // Dequantize to FP32 for reference
            let mut deq = vec![0.0f32; nr * nc];
            let _ = gguf.dequantize_tensor(tensor, &mut deq);
            // Random test input
            let mut rng = rand::thread_rng();
            let x: Vec<f32> = (0..nc).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            // FP32 reference output
            let mut ref_out = vec![0.0f32; nr];
            for i in 0..nr {
                let mut s = 0.0;
                for j in 0..nc { s += deq[i * nc + j] * x[j]; }
                ref_out[i] = s;
            }
            // Q4_K output
            let mut q4_out = vec![0.0f32; nr];
            let raw_ptr = raw.as_ptr() as usize;
            let x_ptr = x.as_ptr() as usize;
            let q4_ptr = q4_out.as_mut_ptr() as usize;
            unsafe {
                let r = std::slice::from_raw_parts(raw_ptr as *const u8, nr * row_bytes);
                let o = std::slice::from_raw_parts_mut(q4_ptr as *mut f32, nr);
                let xi = std::slice::from_raw_parts(x_ptr as *const f32, nc);
                fused_gemv_q4k(r, xi, o, nr, nc);
            }
            // Flag sensitive rows
            let mut sensitive = Vec::new();
            let mut fp32_rows = Vec::new();
            for i in 0..nr {
                let err = (ref_out[i] - q4_out[i]).abs() / (ref_out[i].abs() + 1e-10);
                if err > threshold {
                    sensitive.push(i);
                    fp32_rows.extend_from_slice(&deq[i * nc..(i + 1) * nc]);
                }
            }
            if !sensitive.is_empty() {
                map.insert(tname, (sensitive, fp32_rows));
            }
        }
    }
    map
}

/// Run calibrate_sensitivity with sensible defaults, store in global map.
/// Call once at engine init, e.g. after model load.
pub fn init_sensitivity(gguf: &GgufFile, num_layers: usize) {
    let map = calibrate_row_sensitivity(gguf, num_layers, 0.05);
    let total = map.values().map(|(rows, _)| rows.len()).sum::<usize>();
    if total > 0 {
        set_sensitivity_map(map);
        eprintln!("  Sensitivity: {} rows flagged (FP16 fallback)", total);
    }
}

// =========================================================================
// GEMV (Single input)
// =========================================================================

pub fn forward_linear(
    gguf: &GgufFile,
    tensor: &TensorInfo,
    x: &[f32],
    out: &mut [f32],
    n_threads: usize,
) -> Result<()> {
    let nc = tensor.shape[0] as usize;
    let nr = tensor.shape[1] as usize;
    if x.len() < nc || out.len() < nr {
        bail!("forward_linear shape mismatch");
    }
    let raw = gguf.tensor_raw_bytes(tensor)?;
    match tensor.dtype {
        GgmlDType::Q4_K => {
            // Check sensitivity map for FP32 fallback rows (Idea #4)
            let sensitive = SENSITIVITY_MAP.get().and_then(|m| m.get(tensor.name.as_str()));
            if let Some((srows, fp32_data)) = sensitive {
                // Build a mask: mark which rows are sensitive
                let mut is_sensitive = vec![false; nr];
                for &r in srows { is_sensitive[r] = true; }
                // Split: normal Q4_K rows + sensitive FP32 rows
                let n_blocks = nc / Q4K_BLOCK;
                let row_bytes = n_blocks * Q4K_BYTES;
                let raw_ptr = raw.as_ptr() as usize;
                let x_ptr = x.as_ptr() as usize;
                let out_ptr = out.as_mut_ptr() as usize;
                let fps = fp32_data.as_ptr() as usize;
                (0..n_threads).into_par_iter().for_each(|t| {
                    let rpt = (nr + n_threads - 1) / n_threads;
                    let rs = t * rpt;
                    let re = (rs + rpt).min(nr);
                    if rs >= re { return; }
                    let xi = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, nc) };
                    let o = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, nr) };
                    for i in rs..re {
                        if is_sensitive[i] {
                            // Find which position this row occupies in the sensitive rows list
                            let idx = srows.binary_search(&i).unwrap();
                            let row_start = idx * nc;
                            let w = unsafe { std::slice::from_raw_parts((fps + row_start * 4) as *const f32, nc) };
                            let mut s = 0.0f32;
                            for j in 0..nc { s += w[j] * xi[j]; }
                            o[i] = s;
                        } else {
                            let raw_off = i * row_bytes;
                            let r = unsafe { std::slice::from_raw_parts((raw_ptr + raw_off) as *const u8, row_bytes) };
                            fused_gemv_q4k(r, xi, &mut o[i..i+1], 1, nc);
                        }
                    }
                });
                return Ok(());
            }

            // Normal Q4_K path (no sensitivity map)
            let n_blocks = nc / Q4K_BLOCK;
            let row_bytes = n_blocks * Q4K_BYTES;
            let rpt = (nr + n_threads - 1) / n_threads;
            let raw_ptr = raw.as_ptr() as usize;
            let x_ptr = x.as_ptr() as usize;
            let out_ptr = out.as_mut_ptr() as usize;
            (0..n_threads).into_par_iter().for_each(|t| {
                let rs = t * rpt;
                let re = (rs + rpt).min(nr);
                if rs >= re { return; }
                let rc = re - rs;
                let raw_off = rs * row_bytes;
                let r = unsafe { std::slice::from_raw_parts((raw_ptr + raw_off) as *const u8, rc * row_bytes) };
                let o = unsafe { std::slice::from_raw_parts_mut((out_ptr + rs * 4) as *mut f32, rc) };
                let xi = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, nc) };
                fused_gemv_q4k(r, xi, o, rc, nc);
            });
            Ok(())
        }
        GgmlDType::Q6_K => {
            let n_blocks = nc / 256;
            let row_bytes = n_blocks * 210;
            let rpt = (nr + n_threads - 1) / n_threads;
            let raw_ptr = raw.as_ptr() as usize;
            let x_ptr = x.as_ptr() as usize;
            let out_ptr = out.as_mut_ptr() as usize;
            (0..n_threads).into_par_iter().for_each(|t| {
                let rs = t * rpt;
                let re = (rs + rpt).min(nr);
                if rs >= re { return; }
                let rc = re - rs;
                let raw_off = rs * row_bytes;
                let r = unsafe { std::slice::from_raw_parts((raw_ptr + raw_off) as *const u8, rc * row_bytes) };
                let o = unsafe { std::slice::from_raw_parts_mut((out_ptr + rs * 4) as *mut f32, rc) };
                let xi = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, nc) };
                fused_gemv_q6k(r, xi, o, rc, nc);
            });
            Ok(())
        }
        _ => {
            let mut deq = vec![0.0f32; nr * nc];
            gguf.dequantize_tensor(tensor, &mut deq)?;
            let rpt = (nr + n_threads - 1) / n_threads;
            let out_ptr = out.as_mut_ptr() as usize;
            let deq_ptr = deq.as_ptr() as usize;
            let x_ptr = x.as_ptr() as usize;
            (0..n_threads).into_par_iter().for_each(|t| {
                let rs = t * rpt;
                let re = (rs + rpt).min(nr);
                let deq_i = unsafe { std::slice::from_raw_parts((deq_ptr + rs * nc * 4) as *const f32, (re - rs) * nc) };
                let o = unsafe { std::slice::from_raw_parts_mut((out_ptr + rs * 4) as *mut f32, re - rs) };
                let xi = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, nc) };
                for i in 0..(re - rs) {
                    let mut s = 0.0;
                    for j in 0..nc { s += deq_i[i * nc + j] * xi[j]; }
                    o[i] = s;
                }
            });
            Ok(())
        }
    }
}

// =========================================================================
// Fused Multi-GEMV
// =========================================================================

pub fn forward_linear_multi(
    gguf: &GgufFile,
    tensors: &[&TensorInfo],
    x: &[f32],
    outputs: &mut [&mut [f32]],
    n_threads: usize,
) -> Result<()> {
    for (m, tensor) in tensors.iter().enumerate() {
        let nc = tensor.shape[0] as usize;
        let nr = tensor.shape[1] as usize;
        let raw = gguf.tensor_raw_bytes(tensor)?;
        outputs[m].fill(0.0);
        match tensor.dtype {
            GgmlDType::Q4_K => {
                let n_blocks = nc / Q4K_BLOCK;
                let row_bytes = n_blocks * Q4K_BYTES;
                let rpt = (nr + n_threads - 1) / n_threads;
                let raw_ptr = raw.as_ptr() as usize;
                let x_ptr = x.as_ptr() as usize;
                let out_ptr = outputs[m].as_mut_ptr() as usize;
                (0..n_threads).into_par_iter().for_each(|t| {
                    let rs = t * rpt;
                    let re = (rs + rpt).min(nr);
                    if rs >= re { return; }
                    let rc = re - rs;
                    let raw_off = rs * row_bytes;
                    let r = unsafe { std::slice::from_raw_parts((raw_ptr + raw_off) as *const u8, rc * row_bytes) };
                    let o = unsafe { std::slice::from_raw_parts_mut((out_ptr + rs * 4) as *mut f32, rc) };
                    let xi = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, nc) };
                    fused_gemv_q4k(r, xi, o, rc, nc);
                });
            }
            GgmlDType::Q6_K => {
                let n_blocks = nc / 256;
                let row_bytes = n_blocks * 210;
                let rpt = (nr + n_threads - 1) / n_threads;
                let raw_ptr = raw.as_ptr() as usize;
                let x_ptr = x.as_ptr() as usize;
                let out_ptr = outputs[m].as_mut_ptr() as usize;
                (0..n_threads).into_par_iter().for_each(|t| {
                    let rs = t * rpt;
                    let re = (rs + rpt).min(nr);
                    if rs >= re { return; }
                    let rc = re - rs;
                    let raw_off = rs * row_bytes;
                    let r = unsafe { std::slice::from_raw_parts((raw_ptr + raw_off) as *const u8, rc * row_bytes) };
                    let o = unsafe { std::slice::from_raw_parts_mut((out_ptr + rs * 4) as *mut f32, rc) };
                    let xi = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, nc) };
                    fused_gemv_q6k(r, xi, o, rc, nc);
                });
            }
            _ => {
                let mut deq = vec![0.0f32; nr * nc];
                gguf.dequantize_tensor(tensor, &mut deq)?;
                let rpt = (nr + n_threads - 1) / n_threads;
                let out_ptr = outputs[m].as_mut_ptr() as usize;
                let deq_ptr = deq.as_ptr() as usize;
                let x_ptr = x.as_ptr() as usize;
                (0..n_threads).into_par_iter().for_each(|t| {
                    let rs = t * rpt;
                    let re = (rs + rpt).min(nr);
                    let deq_i = unsafe { std::slice::from_raw_parts((deq_ptr + rs * nc * 4) as *const f32, (re - rs) * nc) };
                    let o = unsafe { std::slice::from_raw_parts_mut((out_ptr + rs * 4) as *mut f32, re - rs) };
                    let xi = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, nc) };
                    for i in 0..(re - rs) {
                        let mut s = 0.0;
                        for j in 0..nc { s += deq_i[i * nc + j] * xi[j]; }
                        o[i] = s;
                    }
                });
            }
        }
    }
    Ok(())
}

// =========================================================================
// Batched GEMM — row-partitioned, block-level weight reuse across batch
// =========================================================================

pub fn forward_linear_batch(
    gguf: &GgufFile,
    tensor: &TensorInfo,
    xs: &[&[f32]],
    outputs: &mut [&mut [f32]],
    n_threads: usize,
) -> Result<()> {
    let bs = xs.len();
    if bs == 0 { return Ok(()); }
    if bs != outputs.len() {
        bail!("forward_linear_batch: {} inputs != {} outputs", bs, outputs.len());
    }
    let nc = tensor.shape[0] as usize;
    let nr = tensor.shape[1] as usize;
    let raw = gguf.tensor_raw_bytes(tensor)?;

    match tensor.dtype {
        GgmlDType::Q4_K => {
            // Batched Q4_K: partition ROWS across threads (block-level weight reuse)
            let n_blocks = nc / Q4K_BLOCK;
            let row_bytes = n_blocks * Q4K_BYTES;
            let rpt = (nr + n_threads - 1) / n_threads;
            let raw_ptr = raw.as_ptr() as usize;

            // Use usize for raw addresses (Sync-safe)
            let x_ptrs_u: Vec<usize> = xs.iter().map(|x| x.as_ptr() as usize).collect();
            let out_ptrs_u: Vec<usize> = outputs.iter_mut().map(|o| o.as_mut_ptr() as usize).collect();

            // Zero outputs before accumulating
            for o in outputs.iter_mut() {
                o.fill(0.0);
            }

            (0..n_threads).into_par_iter().for_each(|t| {
                let rs = t * rpt;
                let re = (rs + rpt).min(nr);
                if rs >= re { return; }
                let rc = re - rs;

                let raw_off = rs * row_bytes;
                let row_raw = unsafe {
                    std::slice::from_raw_parts((raw_ptr + raw_off) as *const u8, rc * row_bytes)
                };

                // Build raw pointer arrays inside the closure
                let local_x_ptrs: Vec<*const f32> = x_ptrs_u.iter()
                    .map(|&p| p as *const f32)
                    .collect();
                let local_out_ptrs: Vec<*mut f32> = out_ptrs_u.iter()
                    .map(|&p| unsafe { (p as *mut f32).add(rs) })
                    .collect();

                // Single batched call: weights loaded ONCE per block, reused for all tokens
                fused_gemv_q4k_batched(&row_raw, &local_x_ptrs, &local_out_ptrs, rc, nc, bs);
            });

            Ok(())
        }
        _ => {
            fallback_gemv(raw, xs, outputs, tensor.dtype, nr, nc, n_threads);
            Ok(())
        }
    }
}

// =========================================================================
// Block dequantization — matches scalar path exactly
// =========================================================================

fn deq_q4k(blk: &[u8], dst: &mut [f32]) {
    let d  = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
    let dm = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
    let mut sc = [0u8; 8]; let mut mn = [0u8; 8];
    for j in 0..4 { sc[j] = blk[4+j] & 63; mn[j] = blk[8+j] & 63; }
    for j in 4..8 { let k=j+4; sc[j]=(blk[k]&0xF)|((blk[j-4]>>6)<<4); mn[j]=(blk[k]>>4)|((blk[j]>>6)<<4); }
    for sb in 0..8 {
        let sv = d * (sc[sb] as f32);
        let mv = dm * (mn[sb] as f32);
        let doff = sb * 32;
        let qo = 16 + sb * 16;
        for i in 0..16 {
            let ql = (blk[qo + i] & 0x0F) as f32;
            let qh = ((blk[qo + i] >> 4) & 0x0F) as f32;
            dst[doff + i * 2]     = sv * ql - mv;
            dst[doff + i * 2 + 1] = sv * qh - mv;
        }
    }
}

// =========================================================================
// Batched Layer GEMV — all tensors in one parallel section
// =========================================================================

struct GemvSpec<'a> {
    raw: &'a [u8],
    x: &'a [f32],
    out: &'a mut [f32],
    n_rows: usize,
    n_cols: usize,
    dtype: GgmlDType,
}

pub fn forward_gemvs(
    gguf: &GgufFile,
    specs: &mut [(&TensorInfo, &[f32], &mut [f32])],
    n_threads: usize,
) -> Result<()> {
    if specs.is_empty() { return Ok(()); }

    let mut gemvs: Vec<GemvSpec> = Vec::with_capacity(specs.len());
    for (tensor, x, out) in specs.iter_mut() {
        let nc = tensor.shape[0] as usize;
        let nr = tensor.shape[1] as usize;
        let raw = gguf.tensor_raw_bytes(tensor)?;
        out.fill(0.0);
        gemvs.push(GemvSpec { raw, x, out, n_rows: nr, n_cols: nc, dtype: tensor.dtype });
    }

    let max_nr = gemvs.iter().map(|g| g.n_rows).max().unwrap_or(1);
    let rpt = (max_nr + n_threads - 1) / n_threads;

    // Convert to raw pointers for Rayon closure
    let mut raw_info: Vec<(usize, usize, usize, usize, usize, bool)> = Vec::with_capacity(gemvs.len());
    for g in &mut gemvs {
        raw_info.push((
            g.raw.as_ptr() as usize,
            g.x.as_ptr() as usize,
            g.out.as_mut_ptr() as usize,
            g.n_rows,
            g.n_cols,
            matches!(g.dtype, GgmlDType::Q4_K),
        ));
    }

    (0..n_threads).into_par_iter().for_each(|t| {
        let rs = t * rpt;
        let re = (rs + rpt).min(max_nr);
        if rs >= re { return; }

        for &(raw_p, x_p, out_p, nr, nc, is_q4k) in &raw_info {
            if rs >= nr { continue; }
            let re_clamped = re.min(nr);
            let rc = re_clamped - rs;
            let n_blocks = nc / 256;
            let row_bytes = n_blocks * if is_q4k { 144 } else { 210 };
            let raw_off = rs * row_bytes;
            let r = unsafe { std::slice::from_raw_parts((raw_p + raw_off) as *const u8, rc * row_bytes) };
            let o = unsafe { std::slice::from_raw_parts_mut((out_p + rs * 4) as *mut f32, rc) };
            let xi = unsafe { std::slice::from_raw_parts(x_p as *const f32, nc) };
            if is_q4k {
                fused_gemv_q4k(r, xi, o, rc, nc);
            } else {
                fused_gemv_q6k(r, xi, o, rc, nc);
            }
        }
    });
    Ok(())
}
// =========================================================================

fn fallback_gemv(raw: &[u8], xs: &[&[f32]], outputs: &mut [&mut [f32]],
                  dtype: GgmlDType, nr: usize, nc: usize, nt: usize) {
    let bs = xs.len();
    let cs = (bs + nt - 1) / nt;
    let xp: Vec<usize> = xs.iter().map(|x| x.as_ptr() as usize).collect();
    let op: Vec<usize> = outputs.iter_mut().map(|o| o.as_mut_ptr() as usize).collect();
    let raw_ptr = raw.as_ptr() as usize;
    let raw_len = raw.len();

    (0..nt).into_par_iter().for_each(|t| {
        let s = t * cs;
        let e = (s + cs).min(bs);
        if s >= e { return; }
        let mut lo = vec![0.0f32; nr];
        let r = unsafe { std::slice::from_raw_parts(raw_ptr as *const u8, raw_len) };

        for b in s..e {
            let x = unsafe { std::slice::from_raw_parts(xp[b] as *const f32, nc) };
            lo.fill(0.0);
            match dtype {
                GgmlDType::Q4_K => fused_gemv_q4k(r, x, &mut lo, nr, nc),
                GgmlDType::Q6_K => fused_gemv_q6k(r, x, &mut lo, nr, nc),
                _ => fused_gemv_q4k(r, x, &mut lo, nr, nc),
            }
            let o = unsafe { std::slice::from_raw_parts_mut(op[b] as *mut f32, nr) };
            for i in 0..nr { o[i] = lo[i]; }
        }
    });
}

// =========================================================================
// Ring GEMV — raw &[u8] slices from per-layer ring buffer
// =========================================================================

/// Batched GEMV dispatch from ring slices. Each spec: (raw, x, out, n_rows, n_cols, block_size).
pub fn forward_gemvs_ring(
    specs: &mut [(&[u8], &[f32], &mut [f32], usize, usize, usize)],
    n_threads: usize,
) {
    if specs.is_empty() { return; }

    let max_nr = specs.iter().map(|s| s.3).max().unwrap_or(1);
    let rpt = (max_nr + n_threads - 1) / n_threads;

    struct RawSpec { raw_p: usize, x_p: usize, out_p: usize, nr: usize, nc: usize, bs: usize }
    let mut rs: Vec<RawSpec> = Vec::with_capacity(specs.len());
    for s in specs.iter_mut() {
        rs.push(RawSpec {
            raw_p: s.0.as_ptr() as usize,
            x_p:   s.1.as_ptr() as usize,
            out_p: s.2.as_mut_ptr() as usize,
            nr:    s.3, nc: s.4, bs: s.5,
        });
    }

    (0..n_threads).into_par_iter().for_each(|t| {
        let row_s = t * rpt;
        let row_e = (row_s + rpt).min(max_nr);
        if row_s >= row_e { return; }

        for spec in &rs {
            if row_s >= spec.nr { continue; }
            let re = row_e.min(spec.nr);
            let rc = re - row_s;
            let n_blocks = spec.nc / Q4K_BLOCK;
            let row_bytes = n_blocks * spec.bs;
            let raw_off = row_s * row_bytes;
            let r = unsafe {
                std::slice::from_raw_parts((spec.raw_p + raw_off) as *const u8, rc * row_bytes)
            };
            let o = unsafe {
                std::slice::from_raw_parts_mut((spec.out_p + row_s * 4) as *mut f32, rc)
            };
            let xi = unsafe {
                std::slice::from_raw_parts(spec.x_p as *const f32, spec.nc)
            };
            if spec.bs == 144 {
                fused_gemv_q4k(r, xi, o, rc, spec.nc);
            } else {
                fused_gemv_q6k(r, xi, o, rc, spec.nc);
            }
        }
    });
}
