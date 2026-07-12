// swamp-engine/src/dspark.rs
// DSPark Cognitivo: speculative decoding com PatternDraft
//
// Algoritmo:
//   1. PatternDraft usa LSH (Locality-Sensitive Hashing) no hidden state
//      para encontrar contextos semanticamente similares no passado
//   2. Retorna tokens de continuacao do melhor match como draft
//   3. DSparkEngine orquestra draft + verificacao via rejection sampling

use rand::Rng;

const DEFAULT_NUM_BITS: usize = 32;
const DEFAULT_CAPACITY: usize = 4096;
const DEFAULT_DRAFT_LENGTH: usize = 3;
const EXACT_MATCH_CONFIDENCE: f32 = 0.90;
const NEAR_MATCH_CONFIDENCE: f32 = 0.50;
const FAR_MATCH_CONFIDENCE: f32 = 0.10;

// =========================================================================
// PatternDraft — draft model baseado em LSH do hidden state
// =========================================================================

pub struct PatternDraft {
    hashes: Vec<u64>,
    tokens: Vec<usize>,
    positions: Vec<usize>,      // absolute KV positions for each observation
    head: usize,
    pub(crate) len: usize,
    capacity: usize,

    lsh_vectors: Vec<Vec<f32>>,
    num_bits: usize,
    embed_dim: usize,

    draft_length: usize,
}

impl PatternDraft {
    pub fn new(embed_dim: usize, num_bits: usize, capacity: usize, draft_length: usize) -> Self {
        let mut rng = rand::thread_rng();
        let lsh_vectors: Vec<Vec<f32>> = (0..num_bits)
            .map(|_| (0..embed_dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect();

        Self {
            hashes: vec![0u64; capacity],
            tokens: vec![0usize; capacity],
            positions: vec![0usize; capacity],
            head: 0,
            len: 0,
            capacity,
            lsh_vectors,
            num_bits,
            embed_dim,
            draft_length,
        }
    }

    pub fn new_default(embed_dim: usize) -> Self {
        Self::new(embed_dim, DEFAULT_NUM_BITS, DEFAULT_CAPACITY, DEFAULT_DRAFT_LENGTH)
    }

    fn ordered_idx(&self, logical: usize) -> Option<usize> {
        if logical >= self.len {
            return None;
        }
        if self.len < self.capacity {
            Some(logical)
        } else {
            Some((self.head + logical) % self.capacity)
        }
    }

    pub fn hash_hidden(&self, hidden: &[f32]) -> u64 {
        let mut hash = 0u64;
        let bits = self.num_bits.min(64);
        for i in 0..bits {
            let dot: f32 = self.lsh_vectors[i]
                .iter()
                .zip(hidden.iter())
                .map(|(v, h)| v * h)
                .sum();
            if dot > 0.0 {
                hash |= 1u64 << i;
            }
        }
        hash
    }

    pub fn observe(&mut self, hidden: &[f32], token: usize) {
        let hash = self.hash_hidden(hidden);
        self.hashes[self.head] = hash;
        self.tokens[self.head] = token;
        self.positions[self.head] = 0; // kept for API compat; use observe_at for position
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Observe with absolute KV position for attention selection.
    pub fn observe_at(&mut self, hidden: &[f32], token: usize, pos: usize) {
        let hash = self.hash_hidden(hidden);
        self.hashes[self.head] = hash;
        self.tokens[self.head] = token;
        self.positions[self.head] = pos;
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Save ring buffer + LSH vectors to disk for cross-session persistence (Idea #3).
    pub fn save_to(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut buf = Vec::with_capacity(4096 + self.capacity * 24);
        // Magic + version
        buf.extend_from_slice(b"SWDP");
        buf.extend_from_slice(&1u32.to_le_bytes());
        // Metadata
        buf.extend_from_slice(&(self.embed_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(self.num_bits as u32).to_le_bytes());
        buf.extend_from_slice(&(self.capacity as u32).to_le_bytes());
        buf.extend_from_slice(&(self.draft_length as u32).to_le_bytes());
        buf.extend_from_slice(&(self.head as u32).to_le_bytes());
        buf.extend_from_slice(&(self.len as u32).to_le_bytes());
        // LSH vectors: num_bits × embed_dim f32
        for v in &self.lsh_vectors {
            for &val in v {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }
        // Ring buffers
        for &h in &self.hashes { buf.extend_from_slice(&h.to_le_bytes()); }
        for &t in &self.tokens { buf.extend_from_slice(&(t as u64).to_le_bytes()); }
        for &p in &self.positions { buf.extend_from_slice(&(p as u64).to_le_bytes()); }
        // Atomic write
        let tmp = format!("{}.tmp", path);
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&buf)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load a previously saved PatternDraft from disk.
    pub fn load_from(path: &str) -> Option<Self> {
        use std::io::Read;
        let data = std::fs::read(path).ok()?;
        if data.len() < 28 { return None; }
        let mut r = std::io::Cursor::new(&data);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf4).ok()?;
        if &buf4 != b"SWDP" { return None; }
        r.read_exact(&mut buf4).ok()?;
        let _version = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4).ok()?; let embed_dim = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4).ok()?; let num_bits = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4).ok()?; let capacity = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4).ok()?; let draft_length = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4).ok()?; let head = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4).ok()?; let len = u32::from_le_bytes(buf4) as usize;
        // LSH vectors
        let mut lsh_vectors = Vec::with_capacity(num_bits);
        for _ in 0..num_bits {
            let mut v = vec![0.0f32; embed_dim];
            for val in v.iter_mut() {
                r.read_exact(&mut buf4).ok()?;
                *val = f32::from_le_bytes(buf4);
            }
            lsh_vectors.push(v);
        }
        // Ring buffers
        let mut hashes = vec![0u64; capacity];
        let mut tokens = vec![0usize; capacity];
        let mut positions = vec![0usize; capacity];
        for h in hashes.iter_mut() {
            r.read_exact(&mut buf8).ok()?;
            *h = u64::from_le_bytes(buf8);
        }
        for t in tokens.iter_mut() {
            r.read_exact(&mut buf8).ok()?;
            *t = u64::from_le_bytes(buf8) as usize;
        }
        for p in positions.iter_mut() {
            r.read_exact(&mut buf8).ok()?;
            *p = u64::from_le_bytes(buf8) as usize;
        }
        Some(Self { hashes, tokens, positions, head, len, capacity, lsh_vectors, num_bits, embed_dim, draft_length })
    }

    pub fn save_if_due(&self, path: &str, observe_count: usize) {
        // Save every 100 observations to avoid excessive I/O
        if observe_count % 100 == 0 && observe_count > 0 {
            let _ = self.save_to(path);
        }
    }

    /// Find up to `n` most similar past positions by LSH Hamming distance.
    /// Returns (absolute_position, distance) sorted by distance ascending.
    /// Useful for selecting cold KV blocks for sparse attention.
    pub fn find_similar_positions(&self, hidden: &[f32], n: usize) -> Vec<(usize, u32)> {
        if self.len < 2 {
            return Vec::new();
        }
        let hash = self.hash_hidden(hidden);
        let search_len = self.len;
        let mut candidates: Vec<(usize, u32)> = (0..search_len)
            .filter_map(|logical| {
                let storage = self.ordered_idx(logical)?;
                let dist = (hash ^ self.hashes[storage]).count_ones();
                Some((self.positions[storage], dist))
            })
            .collect();
        candidates.sort_by_key(|&(_, d)| d);
        candidates.truncate(n);
        candidates
    }

    /// Retorna (draft_tokens, confidence_scores) onde confidence[i] ∈ [0, 1].
    /// Vazio se nao ha dados suficientes ou match muito fraco.
    pub fn draft(&self, hidden: &[f32]) -> (Vec<usize>, Vec<f32>) {
        if self.len < self.draft_length + 1 {
            return (Vec::new(), Vec::new());
        }

        let hash = self.hash_hidden(hidden);
        let search_len = self.len - self.draft_length;

        let mut best_idx: Option<usize> = None;
        let mut best_dist = self.num_bits as u32 + 1;

        for logical in 0..search_len {
            let storage = self.ordered_idx(logical).unwrap();
            let h = self.hashes[storage];
            let dist = (hash ^ h).count_ones();
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(logical);
            }
        }

        if let Some(logical_match) = best_idx {
            let conf = if best_dist == 0 {
                EXACT_MATCH_CONFIDENCE
            } else if best_dist <= (self.num_bits as u32) / 4 {
                NEAR_MATCH_CONFIDENCE
            } else if best_dist <= (self.num_bits as u32) / 2 {
                FAR_MATCH_CONFIDENCE
            } else {
                return (Vec::new(), Vec::new());
            };

            let mut tokens = Vec::with_capacity(self.draft_length);
            let mut confs = Vec::with_capacity(self.draft_length);

            for j in 1..=self.draft_length {
                if let Some(storage) = self.ordered_idx(logical_match + j) {
                    tokens.push(self.tokens[storage]);
                    confs.push(conf);
                }
            }

            (tokens, confs)
        } else {
            (Vec::new(), Vec::new())
        }
    }
}

// =========================================================================
// LogitCache — cache de logits para acceptance boosting
// =========================================================================
//
// Mapeia (lsh_hash, token_id) → logits. Se o mesmo par aparece de novo
// dentro da janela de especulação, pula o forward pass (logits) e reusa
// o resultado anterior.

const LOGIT_CACHE_CAPACITY: usize = 8;

pub struct LogitCache {
    hashes: [u64; LOGIT_CACHE_CAPACITY],
    tokens: [usize; LOGIT_CACHE_CAPACITY],
    logits: [Option<Vec<f32>>; LOGIT_CACHE_CAPACITY],
    head: usize,
    len: usize,
}

impl LogitCache {
    pub fn new() -> Self {
        Self {
            hashes: [0u64; LOGIT_CACHE_CAPACITY],
            tokens: [0usize; LOGIT_CACHE_CAPACITY],
            logits: [const { None }; LOGIT_CACHE_CAPACITY],
            head: 0,
            len: 0,
        }
    }

    pub fn lookup(&self, hash: u64, token: usize) -> Option<&[f32]> {
        for i in 0..self.len {
            if self.hashes[i] == hash && self.tokens[i] == token {
                if let Some(ref l) = self.logits[i] {
                    return Some(l.as_slice());
                }
            }
        }
        None
    }

    pub fn insert(&mut self, hash: u64, token: usize, logits: Vec<f32>) {
        self.hashes[self.head] = hash;
        self.tokens[self.head] = token;
        self.logits[self.head] = Some(logits);
        self.head = (self.head + 1) % LOGIT_CACHE_CAPACITY;
        if self.len < LOGIT_CACHE_CAPACITY {
            self.len += 1;
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
        self.head = 0;
    }
}

// =========================================================================
// DSparkEngine — orquestra draft + verificação
// =========================================================================

pub struct DSparkEngine {
    pub draft_model: PatternDraft,
    pub logit_cache: LogitCache,
    pub max_draft: usize,
    pub min_acceptance_prob: f64,
    stats: DSparkStats,
}

#[derive(Debug, Default, Clone)]
pub struct DSparkStats {
    pub total_steps: u64,
    pub total_draft_tokens: u64,
    pub accepted_tokens: u64,
    pub rejected_steps: u64,
    pub total_draft_attempts: u64,
    pub draft_accepted: u64,
    pub draft_rejected: u64,
    pub total_rollbacks: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl DSparkEngine {
    pub fn new(embed_dim: usize) -> Self {
        Self {
            draft_model: PatternDraft::new_default(embed_dim),
            logit_cache: LogitCache::new(),
            max_draft: DEFAULT_DRAFT_LENGTH,
            min_acceptance_prob: 0.01,
            stats: DSparkStats::default(),
        }
    }

    pub fn with_params(embed_dim: usize, num_bits: usize, capacity: usize, draft_length: usize) -> Self {
        Self {
            draft_model: PatternDraft::new(embed_dim, num_bits, capacity, draft_length),
            logit_cache: LogitCache::new(),
            max_draft: draft_length,
            min_acceptance_prob: 0.01,
            stats: DSparkStats::default(),
        }
    }

    /// Load draft model from disk if available (Idea #3 — cross-session persistence).
    pub fn load_draft_cache(&mut self, path: &str) {
        if let Some(draft) = PatternDraft::load_from(path) {
            self.draft_model = draft;
        }
    }

    /// Save draft model to disk for future sessions.
    pub fn save_draft_cache(&self, path: &str) -> std::io::Result<()> {
        self.draft_model.save_to(path)
    }

    pub fn stats(&self) -> &DSparkStats {
        &self.stats
    }

    pub fn stats_mut(&mut self) -> &mut DSparkStats {
        &mut self.stats
    }

    /// Limpa o cache de logits (chamar no inicio de cada decode step).
    pub fn clear_logit_cache(&mut self) {
        self.logit_cache.clear();
    }

    pub fn logit_cache_hit(&mut self) {
        self.stats.cache_hits += 1;
    }

    pub fn logit_cache_miss(&mut self) {
        self.stats.cache_misses += 1;
    }

    /// Observa um hidden state + token gerado para alimentar o PatternDraft.
    pub fn observe(&mut self, hidden: &[f32], token: usize) {
        self.draft_model.observe(hidden, token);
    }

    /// Observa com posicao absoluta no KV cache (para seleção de blocos frios).
    pub fn observe_at(&mut self, hidden: &[f32], token: usize, pos: usize) {
        self.draft_model.observe_at(hidden, token, pos);
    }

    /// Retorna ate `n` posicoes do KV cache mais similares ao hidden state atual.
    pub fn find_attention_candidates(&self, hidden: &[f32], n: usize) -> Vec<(usize, u32)> {
        self.draft_model.find_similar_positions(hidden, n)
    }

    /// Prediz quais páginas do KV cache serão acessadas em breve, baseado em
    /// padrões LSH do histórico. Retorna flat page IDs para prefetch.
    ///
    /// Para cada match LSH, retorna a faixa de páginas ao redor da posição
    /// de match (window de `lookback` + `lookahead` tokens), expandindo para
    /// o flat page ID via `page_id_fn`.
    /// Útil para DSPark-guided cold prefetch: carregar páginas frias antes
    /// do acesso real durante a atenção.
    pub fn predict_cold_pages(
        &self,
        hidden: &[f32],
        current_pos: usize,
        n_matches: usize,
        lookback: usize,
        lookahead: usize,
        page_id_fn: impl Fn(usize) -> usize,
    ) -> Vec<usize> {
        let matches = self.find_attention_candidates(hidden, n_matches);
        if matches.is_empty() {
            return Vec::new();
        }
        let mut pages: Vec<usize> = Vec::new();
        for &(match_pos, _dist) in &matches {
            // Only consider matches that are far enough back to be useful
            // (skip matches near current position — those pages are already hot)
            if match_pos >= current_pos || current_pos - match_pos < 128 {
                continue;
            }
            let start = match_pos.saturating_sub(lookback);
            let end = (match_pos + lookahead).min(current_pos);
            let start_pid = page_id_fn(start);
            let end_pid = page_id_fn(end);
            for pid in start_pid..=end_pid {
                if !pages.contains(&pid) {
                    pages.push(pid);
                }
            }
        }
        pages.truncate(64); // cap at 64 pages per step to avoid I/O storms
        pages
    }

    /// Tenta speculative decoding para um passo.
    ///
    /// `hidden` — hidden state RMSNormado (antes do output projection)
    /// `target_logits` — logits do passo atual (para amostrar t0)
    /// `sample_token` — amostra um token de logits
    /// `verify_fn` — executa forward pass com um token draft e retorna (logits, token_amostrado)
    ///
    /// Retorna lista de tokens aceitos (pelo menos 1).
    pub fn speculate(
        &mut self,
        hidden: &[f32],
        target_logits: &[f32],
        sample_token: impl Fn(&[f32]) -> usize,
        verify_fn: impl Fn(usize) -> (Vec<f32>, usize),
    ) -> Vec<usize> {
        self.stats.total_steps += 1;

        // 1. Amostra o token normal da distribuicao target
        let t0 = sample_token(target_logits);
        let mut accepted = vec![t0];
        self.stats.accepted_tokens += 1;

        // 2. Obtem draft tokens + confiancas do PatternDraft
        let (draft_tokens, draft_confs) = self.draft_model.draft(hidden);
        if draft_tokens.is_empty() {
            return accepted;
        }

        // 3. Verifica cada token draft com rejection sampling
        for (i, d_token) in draft_tokens.iter().enumerate() {
            let d_conf = draft_confs[i] as f64;
            self.stats.total_draft_tokens += 1;

            // Executa forward pass do target com o token draft
            let (logits, _sampled) = verify_fn(*d_token);

            // Probabilidade do token draft na distribuicao target
            let target_p = softmax_at(logits.as_slice(), *d_token);
            if target_p <= 0.0 {
                self.stats.rejected_steps += 1;
                break;
            }

            let target_lp = (target_p as f64).ln();
            let draft_lp = (d_conf.max(1e-30)).ln();
            let ratio = (target_lp - draft_lp).exp().min(1.0);

            let r: f64 = fast_rand();
            if r < ratio {
                accepted.push(*d_token);
                self.stats.accepted_tokens += 1;
                self.stats.draft_accepted += 1;
            } else {
                let resampled = resample_from_corrected(&logits, *d_token, target_p, &sample_token);
                accepted.push(resampled);
                self.stats.draft_rejected += 1;
                self.stats.rejected_steps += 1;
                break;
            }
        }

        accepted
    }
}

// =========================================================================
// Helpers
// =========================================================================

pub(crate) fn softmax_at(logits: &[f32], token: usize) -> f32 {
    if token >= logits.len() { return 0.0; }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    let mut t_exp = 0.0f64;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l as f64 - max_val as f64).exp();
        sum += e;
        if i == token {
            t_exp = e;
        }
    }
    if sum <= 0.0 { return 0.0; }
    (t_exp / sum) as f32
}

pub(crate) fn resample_from_corrected(
    logits: &[f32],
    _draft_token: usize,
    _target_p_draft: f32,
    sample_token: impl Fn(&[f32]) -> usize,
) -> usize {
    sample_token(logits)
}

pub(crate) fn fast_rand() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = const { Cell::new(0x123456789abcdef) };
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x as f64) / (u64::MAX as f64)
    })
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_draft_basic() {
        let embed_dim = 64;
        let mut draft = PatternDraft::new(embed_dim, 16, 128, 3);
        let h = vec![0.5f32; embed_dim];

        // Observa uma sequencia de estados + tokens
        for i in 0..20 {
            let h_i: Vec<f32> = (0..embed_dim).map(|j| (i as f32 * 0.1) + (j as f32 * 0.01)).collect();
            draft.observe(&h_i, i % 100);
        }

        // Busca um estado similar ao que ja foi observado
        let h_query: Vec<f32> = (0..embed_dim).map(|j| (5.0 * 0.1) + (j as f32 * 0.01)).collect();
        let (tokens, confs) = draft.draft(&h_query);
        assert!(!tokens.is_empty(), "should find a match");
        assert_eq!(tokens.len(), confs.len(), "tokens and confs must match");
    }

    #[test]
    fn test_pattern_draft_empty() {
        let draft = PatternDraft::new(64, 16, 128, 3);
        let h = vec![0.5f32; 64];
        let (tokens, _) = draft.draft(&h);
        assert!(tokens.is_empty(), "empty store should return empty draft");
    }

    #[test]
    fn test_hash_hidden_deterministic() {
        let draft = PatternDraft::new(64, 16, 128, 3);
        let h = vec![0.5f32; 64];
        let h1 = draft.hash_hidden(&h);
        let h2 = draft.hash_hidden(&h);
        assert_eq!(h1, h2, "hash must be deterministic");
    }

    #[test]
    fn test_dspark_speculate_no_rejection() {
        let embed_dim = 64;
        let mut engine = DSparkEngine::new(embed_dim);

        // Preenche o padrao
        let h = vec![1.0f32; embed_dim];
        for i in 0..10 {
            let hi: Vec<f32> = (0..embed_dim).map(|j| (i as f32 * 0.1) + (j as f32 * 0.01)).collect();
            engine.observe(&hi, 42 + i);
        }

        let target_logits: Vec<f32> = (0..100).map(|i| if i == 3 { 10.0 } else { 0.0 }).collect();
        let result = engine.speculate(
            &h,
            &target_logits,
            |_| 3,
            |d| {
                let logits: Vec<f32> = (0..100).map(|i| if i == d || i == 3 { 10.0 } else { 0.0 }).collect();
                (logits, d)
            },
        );
        assert!(result.len() >= 1, "should accept at least 1 token");
    }

    #[test]
    fn test_fast_rand_range() {
        for _ in 0..1000 {
            let r = fast_rand();
            assert!(r >= 0.0 && r < 1.0, "fast_rand out of range: {}", r);
        }
    }
}
