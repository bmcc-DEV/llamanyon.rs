// swamp-engine/src/model.rs
// Model: carregador de tensores e metadados de GGUF do modelo

use swamp_gguf::GgufFile;
use std::path::Path;
use anyhow::{Result, Context, anyhow};
use std::sync::OnceLock;
use std::alloc::{alloc_zeroed, dealloc, Layout};

pub struct ModelConfig {
    pub architecture: String,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub embed_dim: usize,
    pub context_len: usize,
    pub vocab_size: usize,
    pub rope_freq_base: f32,
    pub rope_dim: usize,
}

/// Per-layer ring buffer: contiguous byte slices for all 7 GEMV tensors.
pub struct RingView {
    pub ring: Vec<u8>,
    // Offsets into ring for each tensor
    pub q_off: usize, pub q_nr: usize, pub q_nc: usize, pub q_bs: usize,
    pub k_off: usize, pub k_nr: usize, pub k_nc: usize, pub k_bs: usize,
    pub v_off: usize, pub v_nr: usize, pub v_nc: usize, pub v_bs: usize,
    pub o_off: usize, pub o_nr: usize, pub o_nc: usize, pub o_bs: usize,
    pub gate_off: usize, pub gate_nr: usize, pub gate_nc: usize, pub gate_bs: usize,
    pub up_off: usize,   pub up_nr: usize,   pub up_nc: usize,   pub up_bs: usize,
    pub down_off: usize, pub down_nr: usize, pub down_nc: usize, pub down_bs: usize,
    // Pre-computed row strides
    q_row_stride: usize, k_row_stride: usize, v_row_stride: usize, o_row_stride: usize,
    gate_row_stride: usize, up_row_stride: usize, down_row_stride: usize,
}

fn block_size_for_dtype(dtype: swamp_gguf::GgmlDType) -> usize {
    match dtype {
        swamp_gguf::GgmlDType::Q4_K => 144,
        swamp_gguf::GgmlDType::Q6_K => 210,
        _ => 144,
    }
}

fn build_layer_ring(gguf: &GgufFile, layer: usize) -> Result<RingView> {
    let names = [
        format!("blk.{}.attn_q.weight", layer),
        format!("blk.{}.attn_k.weight", layer),
        format!("blk.{}.attn_v.weight", layer),
        format!("blk.{}.attn_output.weight", layer),
        format!("blk.{}.ffn_gate.weight", layer),
        format!("blk.{}.ffn_up.weight", layer),
        format!("blk.{}.ffn_down.weight", layer),
    ];
    // Get all tensor metadata
    let meta: Vec<_> = names.iter().map(|n| {
        let t = gguf.tensor_or_err(n).map_err(|e| anyhow!("{n}: {e}"))?;
        let n_rows = t.shape[1] as usize; // output dim
        let n_cols = t.shape[0] as usize; // input dim
        let bs = block_size_for_dtype(t.dtype);
        let raw = gguf.tensor_raw_bytes(t)?;
        Ok((n_rows, n_cols, bs, raw.to_vec()))
    }).collect::<Result<Vec<_>>>()?;

    let mut ring = Vec::new();
    let mut offsets = Vec::with_capacity(7);
    for (nr, nc, bs, data) in &meta {
        offsets.push(ring.len());
        ring.extend_from_slice(data);
    }
    // Pad ring to cache-line boundary
    while ring.len() % 64 != 0 { ring.push(0); }

    Ok(RingView {
        ring,
        q_off: offsets[0], q_nr: meta[0].0, q_nc: meta[0].1, q_bs: meta[0].2,
        k_off: offsets[1], k_nr: meta[1].0, k_nc: meta[1].1, k_bs: meta[1].2,
        v_off: offsets[2], v_nr: meta[2].0, v_nc: meta[2].1, v_bs: meta[2].2,
        o_off: offsets[3], o_nr: meta[3].0, o_nc: meta[3].1, o_bs: meta[3].2,
        gate_off: offsets[4], gate_nr: meta[4].0, gate_nc: meta[4].1, gate_bs: meta[4].2,
        up_off: offsets[5],   up_nr: meta[5].0,   up_nc: meta[5].1,   up_bs: meta[5].2,
        down_off: offsets[6], down_nr: meta[6].0, down_nc: meta[6].1, down_bs: meta[6].2,
        q_row_stride: (meta[0].1 / 256) * meta[0].2,
        k_row_stride: (meta[1].1 / 256) * meta[1].2,
        v_row_stride: (meta[2].1 / 256) * meta[2].2,
        o_row_stride: (meta[3].1 / 256) * meta[3].2,
        gate_row_stride: (meta[4].1 / 256) * meta[4].2,
        up_row_stride: (meta[5].1 / 256) * meta[5].2,
        down_row_stride: (meta[6].1 / 256) * meta[6].2,
    })
}

/// A group of consecutive layers that share identical weight tensors (all 7).
/// Layers within a group can be processed with a single weight load per tensor.
pub type LayerGroup = Vec<usize>;

/// Detect which consecutive layers share fully identical weight tensors
/// by comparing raw ring bytes of each of the 7 tensor types.
fn detect_weight_groups(layer_rings: &[RingView], num_layers: usize) -> Vec<LayerGroup> {
    if num_layers == 0 {
        return Vec::new();
    }
    let mut groups: Vec<LayerGroup> = Vec::new();
    let mut current: LayerGroup = vec![0];

    for l in 1..num_layers {
        let prev = &layer_rings[l - 1];
        let cur = &layer_rings[l];
        if prev.ring.len() == cur.ring.len() && prev.ring == cur.ring {
            current.push(l);
        } else {
            groups.push(current);
            current = vec![l];
        }
    }
    groups.push(current);
    groups
}

pub struct Model {
    pub gguf: GgufFile,
    pub config: ModelConfig,
    pub layer_rings: Vec<RingView>,
    /// Groups of consecutive layers that share identical weight tensors.
    /// Each group lists the layer indices in order; size=1 means no sharing.
    pub shared_groups: Vec<LayerGroup>,
}

impl RingView {
    #[inline] pub fn q_slice(&self) -> &[u8] { &self.ring[self.q_off..self.k_off] }
    #[inline] pub fn k_slice(&self) -> &[u8] { &self.ring[self.k_off..self.v_off] }
    #[inline] pub fn v_slice(&self) -> &[u8] { &self.ring[self.v_off..self.o_off] }
    #[inline] pub fn o_slice(&self) -> &[u8] { &self.ring[self.o_off..self.gate_off] }
    #[inline] pub fn gate_slice(&self) -> &[u8] { &self.ring[self.gate_off..self.up_off] }
    #[inline] pub fn up_slice(&self) -> &[u8] { &self.ring[self.up_off..self.down_off] }
    #[inline] pub fn down_slice(&self) -> &[u8] { &self.ring[self.down_off..] }
}

impl Model {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let gguf = GgufFile::open(path.as_ref())
            .with_context(|| format!("Falha ao abrir o GGUF em: {:?}", path.as_ref()))?;

        let md = &gguf.metadata;

        // Extrai configuracoes usando o compat automatizado
        let architecture = gguf.architecture().to_string();
        let num_layers = md.n_layer()? as usize;
        let num_heads = md.n_head()? as usize;
        let num_kv_heads = md.n_kv_head() as usize;
        let embed_dim = md.n_embd()? as usize;
        let context_len = md.n_ctx_train() as usize;
        let rope_freq_base = md.rope_freq_base();
        let rope_dim = md.rope_dimension() as usize;

        // Obtem vocab_size do array de tokens
        let vocab_size = md.get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_array())
            .map(|arr| arr.len())
            .unwrap_or(32000); // fallback padrao

        let config = ModelConfig {
            architecture,
            num_layers,
            num_heads,
            num_kv_heads,
            embed_dim,
            context_len,
            vocab_size,
            rope_freq_base,
            rope_dim,
        };

    // Build ring buffers for GEMV dispatch
    // Pre-allocate total ring size to force dual-channel physical pages (flex mode: first 16GB)
    let total_ring_bytes: usize = (0..num_layers).map(|l| {
        let names = [
            format!("blk.{l}.attn_q.weight"), format!("blk.{l}.attn_k.weight"), format!("blk.{l}.attn_v.weight"),
            format!("blk.{l}.attn_output.weight"), format!("blk.{l}.ffn_gate.weight"),
            format!("blk.{l}.ffn_up.weight"), format!("blk.{l}.ffn_down.weight"),
        ];
        names.iter().map(|n| {
            let t = gguf.tensor_or_err(n).ok()?;
            let raw = gguf.tensor_raw_bytes(t).ok()?;
            Some(raw.len())
        }).filter_map(|x| x).sum::<usize>()
    }).sum();

    // Pre-fault a huge mapping to lock physical pages in dual-channel region
    let pre_alloc_size = (total_ring_bytes + 64 * 1024 * 1024).max(512 * 1024 * 1024);
    let pre_layout = Layout::from_size_align(pre_alloc_size, 4096).unwrap();
    let pre_ptr = unsafe { alloc_zeroed(pre_layout) } as *mut u8;
    if !pre_ptr.is_null() {
        // Touch every page to force physical allocation in low addresses (dual-channel)
        for off in (0..pre_alloc_size).step_by(4096) {
            unsafe { std::ptr::write_volatile(pre_ptr.add(off), 0u8); }
        }
        // Hint huge pages
        unsafe { libc::madvise(pre_ptr as *mut libc::c_void, pre_alloc_size, libc::MADV_HUGEPAGE); }
    }

    let mut layer_rings = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        layer_rings.push(build_layer_ring(&gguf, l)?);
    }

    // Free pre-alloc after rings are built (now rings own physical pages nearby)
    if !pre_ptr.is_null() {
        unsafe { dealloc(pre_ptr as *mut u8, pre_layout); }
    }

    println!("  Rings built: {} layers ({} MB)", layer_rings.len(), layer_rings.iter().map(|r| r.ring.len()).sum::<usize>() / (1024*1024));

    // Detect cross-layer weight sharing
    let shared_groups = detect_weight_groups(&layer_rings, num_layers);
    let shared_layers: usize = shared_groups.iter().filter(|g| g.len() > 1).map(|g| g.len()).sum();
    if shared_layers > 0 {
        println!("  Weight sharing: {} layers in {} groups", shared_layers, shared_groups.iter().filter(|g| g.len() > 1).count());
    }

        Ok(Self { gguf, config, layer_rings, shared_groups })
    }

    pub fn print_info(&self) {
        println!("=== Configurações do Model ===");
        println!("  Arquitetura:   {}", self.config.architecture);
        println!("  Camadas:       {}", self.config.num_layers);
        println!("  Embedding:     {}", self.config.embed_dim);
        println!("  Heads:         {} (KV: {})", self.config.num_heads, self.config.num_kv_heads);
        println!("  Contexto:      {}", self.config.context_len);
        println!("  RoPE freq:     {}", self.config.rope_freq_base);
        println!("  RoPE dim:      {}", self.config.rope_dim);
        println!("  Vocab size:    {}", self.config.vocab_size);
    }
}
