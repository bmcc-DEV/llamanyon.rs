use crate::model::Model;
use anyhow::Result;
use std::sync::Mutex;
use swamp_kernels::fused_gemv_q4k::unpack_scales_q4k;

#[derive(Clone)]
struct BlockStats {
    act_sum: [f64; 256],
    count: u64,
}

impl Default for BlockStats {
    fn default() -> Self { Self { act_sum: [0.0; 256], count: 0 } }
}

pub struct TensorStats {
    pub blocks: Vec<Mutex<BlockStats>>,
}

pub struct QatCalibrator {
    pub layer_stats: Vec<[TensorStats; 7]>,
    n_blocks_embed: usize,
    n_blocks_ffn: usize,
}

impl QatCalibrator {
    pub fn new(model: &Model) -> Self {
        Self::new_with_layers(model, model.config.num_layers)
    }

    pub fn new_with_layers(model: &Model, max_layers: usize) -> Self {
        let num_layers = model.config.num_layers.min(max_layers);
        let embed_dim = model.config.embed_dim;
        let n_blocks_embed = embed_dim / 256;
        // Compute ffn_dim from the first layer's ring
        let n_blocks_ffn = if num_layers > 0 {
            model.layer_rings[0].n_blocks_down
        } else { 1 };

        let layer_stats = (0..num_layers).map(|l| {
            let r = &model.layer_rings[l];
            [
                TensorStats { blocks: (0..r.q_nr * n_blocks_embed).map(|_| Mutex::new(BlockStats::default())).collect() },
                TensorStats { blocks: (0..r.k_nr * n_blocks_embed).map(|_| Mutex::new(BlockStats::default())).collect() },
                TensorStats { blocks: (0..r.v_nr * n_blocks_embed).map(|_| Mutex::new(BlockStats::default())).collect() },
                TensorStats { blocks: (0..r.o_nr * n_blocks_embed).map(|_| Mutex::new(BlockStats::default())).collect() },
                TensorStats { blocks: (0..r.gate_nr * n_blocks_embed).map(|_| Mutex::new(BlockStats::default())).collect() },
                TensorStats { blocks: (0..r.up_nr * n_blocks_embed).map(|_| Mutex::new(BlockStats::default())).collect() },
                TensorStats { blocks: (0..r.down_nr * n_blocks_ffn).map(|_| Mutex::new(BlockStats::default())).collect() },
            ]
        }).collect();

        Self { layer_stats, n_blocks_embed, n_blocks_ffn }
    }

    pub const T_Q: usize = 0;
    pub const T_K: usize = 1;
    pub const T_V: usize = 2;
    pub const T_O: usize = 3;
    pub const T_GATE: usize = 4;
    pub const T_UP: usize = 5;
    pub const T_DOWN: usize = 6;

    pub fn record_activation(
        &self, layer: usize, tensor_idx: usize, block_idx: usize, x_blk: &[f32]
    ) {
        if layer >= self.layer_stats.len() || tensor_idx >= 7 { return; }
        let ts = &self.layer_stats[layer][tensor_idx];
        if block_idx >= ts.blocks.len() { return; }
        let mut stats = ts.blocks[block_idx].lock().unwrap();
        let n = 256.min(x_blk.len());
        for i in 0..n { stats.act_sum[i] += x_blk[i].abs() as f64; }
        stats.count += 1;
    }

    #[inline]
    pub fn block_idx(row: usize, bc: usize, n_blocks_per_row: usize) -> usize {
        row * n_blocks_per_row + bc
    }

    pub fn apply_ring(&self, model: &mut Model) -> Result<u64> {
        let start = std::time::Instant::now();
        let mut total_updated = 0u64;

        for (l, ring) in model.layer_rings.iter_mut().enumerate() {
            if l >= self.layer_stats.len() { break; }
            let tl = &self.layer_stats[l];
            let tensors: [(usize, usize, usize, usize, usize); 7] = [
                (0, ring.q_off,   ring.k_off,   ring.q_nr,   self.n_blocks_embed),
                (1, ring.k_off,   ring.v_off,   ring.k_nr,   self.n_blocks_embed),
                (2, ring.v_off,   ring.o_off,   ring.v_nr,   self.n_blocks_embed),
                (3, ring.o_off,   ring.gate_off, ring.o_nr,  self.n_blocks_embed),
                (4, ring.gate_off, ring.up_off,  ring.gate_nr, self.n_blocks_embed),
                (5, ring.up_off,  ring.down_off, ring.up_nr,  self.n_blocks_embed),
                (6, ring.down_off, ring.data.len(), ring.down_nr, self.n_blocks_ffn),
            ];

            for &(ti, start_off, end_off, n_rows, n_blocks) in &tensors {
                let ts = &tl[ti];
                let row_bytes = n_blocks * 144;
                let data = &mut ring.data[start_off..end_off];

                for row in 0..n_rows {
                    for bc in 0..n_blocks {
                        let off = row * row_bytes + bc * 144;
                        if off + 144 > data.len() { continue; }
                        let bi = row * n_blocks + bc;
                        if bi >= ts.blocks.len() { continue; }

                        let stats = ts.blocks[bi].lock().unwrap();
                        if stats.count < 1 { continue; }

                        let raw: &[u8] = &data[off..off + 144];
                        let d_cur = half::f16::from_le_bytes([raw[0], raw[1]]).to_f32() as f64;
                        let dm_cur = half::f16::from_le_bytes([raw[2], raw[3]]).to_f32() as f64;
                        let (scales, mins) = unpack_scales_q4k(&raw[4..16]);
                        let qs: &[u8] = &raw[16..144];

                        let mut target = [0.0f64; 256];
                        for sb in 0..8 {
                            let sv = d_cur * (scales[sb] as f64);
                            let mv = dm_cur * (mins[sb] as f64);
                            for i in 0..16 {
                                let ql = (qs[sb * 16 + i] & 0x0F) as f64;
                                let qh = ((qs[sb * 16 + i] >> 4) & 0x0F) as f64;
                                target[sb * 32 + i * 2]     = sv * ql - mv;
                                target[sb * 32 + i * 2 + 1] = sv * qh - mv;
                            }
                        }

                        let total_act = stats.act_sum.iter().sum::<f64>();
                        if total_act < 1e-10 { continue; }
                        let mut imp = [0.0f64; 256];
                        for i in 0..256 { imp[i] = stats.act_sum[i] / total_act; }

                        let dc = [d_cur * 0.997, d_cur * 0.999, d_cur, d_cur * 1.001, d_cur * 1.003];
                        let dmc = [dm_cur * 0.997, dm_cur * 0.999, dm_cur, dm_cur * 1.001, dm_cur * 1.003];
                        let (bd, bdm) = Self::gs(&target, &imp, qs, &scales, &mins, &dc, &dmc);
                        if bd != d_cur || bdm != dm_cur {
                            data[off..off + 2].copy_from_slice(&half::f16::from_f32(bd as f32).to_le_bytes());
                            data[off + 2..off + 4].copy_from_slice(&half::f16::from_f32(bdm as f32).to_le_bytes());
                            total_updated += 1;
                        }
                    }
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        tracing::info!("QAT: {} blocks updated in {:.2}s", total_updated, elapsed);
        Ok(total_updated)
    }

    fn gs(
        target: &[f64; 256], imp: &[f64; 256], qs: &[u8],
        scales: &[u8; 8], mins: &[u8; 8],
        dc: &[f64; 5], dmc: &[f64; 5],
    ) -> (f64, f64) {
        let mut bd = dc[2]; let mut bdm = dmc[2]; let mut bm = f64::MAX;
        for &d in dc { for &dm in dmc {
            let mut m = 0.0f64;
            for sb in 0..8 {
                let sv = d * (scales[sb] as f64); let mv = dm * (mins[sb] as f64);
                for i in 0..16 {
                    let ql = (qs[sb * 16 + i] & 0x0F) as f64;
                    let qh = ((qs[sb * 16 + i] >> 4) & 0x0F) as f64;
                    let ie = sb * 32 + i * 2;
                    let io = ie + 1;
                    let ee = target[ie] - (sv * ql - mv);
                    let eo = target[io] - (sv * qh - mv);
                    m += (ee * ee) * imp[ie] + (eo * eo) * imp[io];
                }
            }
            if m < bm { bm = m; bd = d; bdm = dm; }
        }}
        (bd, bdm)
    }
}
