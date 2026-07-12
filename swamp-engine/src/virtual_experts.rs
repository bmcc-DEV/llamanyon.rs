// swamp-engine/src/virtual_experts.rs
// MoE Expert Prefetch — divide as linhas do FFN Gate/Up em "experts virtuais"
// e usa o hash LSH do DSPark para predizer qual expert será ativado.
// O executor usa a predição para pré-carregar (madvise WILLNEED) aquelas
// linhas de TODAS as camadas antes do FFN GEMV do próximo token.

pub struct VirtualExpertPrefetcher {
    num_experts: usize,
    ffn_rows: usize,
    expert_row_ranges: Vec<(usize, usize)>,
    last_predicted: Option<usize>,
    hits: u64,
    total: u64,
}

impl VirtualExpertPrefetcher {
    pub fn new(num_experts: usize, ffn_rows: usize) -> Self {
        let rows_per_expert = ffn_rows / num_experts;
        let mut expert_row_ranges = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            let start = i * rows_per_expert;
            let end = if i == num_experts - 1 {
                ffn_rows
            } else {
                (i + 1) * rows_per_expert
            };
            expert_row_ranges.push((start, end));
        }
        Self {
            num_experts,
            ffn_rows,
            expert_row_ranges,
            last_predicted: None,
            hits: 0,
            total: 0,
        }
    }

    /// Map LSH hash → expert ID
    pub fn hash_to_expert(&self, hash: u64) -> usize {
        (hash % self.num_experts as u64) as usize
    }

    /// Given an LSH hash, return (start_row, end_row) for the predicted expert.
    pub fn predict_expert(&mut self, hash: u64) -> (usize, usize) {
        let expert = self.hash_to_expert(hash);
        self.total += 1;
        if self.last_predicted == Some(expert) {
            self.hits += 1;
        } else if self.total == 1 {
            // first prediction — don't count as miss or hit
        }
        self.last_predicted = Some(expert);
        self.expert_row_ranges[expert]
    }

    pub fn accuracy(&self) -> f64 {
        if self.total <= 1 {
            return 0.0;
        }
        self.hits as f64 / (self.total - 1) as f64
    }
}
