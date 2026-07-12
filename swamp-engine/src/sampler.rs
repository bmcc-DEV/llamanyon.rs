// swamp-engine/src/sampler.rs
// Sampler: amostragem de tokens (top-k, top-p, temperatura)

use rand::Rng;

pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
        }
    }

    /// Executa amostragem probabilistica a partir de um array de logits
    pub fn sample(&self, logits: &[f32]) -> usize {
        if logits.is_empty() {
            return 0;
        }

        let mut rng = rand::thread_rng();

        // 1. Aplica temperatura e calcula exponenciais (probs)
        let temp = self.temperature.max(1e-6);
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        let mut probs: Vec<(usize, f32)> = logits.iter()
            .enumerate()
            .map(|(idx, &l)| {
                let p = ((l - max_logit) / temp).exp();
                (idx, p)
            })
            .collect();

        // Soma total para normalizacao
        let sum_prob: f32 = probs.iter().map(|&(_, p)| p).sum();
        for item in &mut probs {
            item.1 /= sum_prob;
        }

        // 2. Ordena decrescente por probabilidade
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 3. Aplica Top-K
        if self.top_k > 0 && self.top_k < probs.len() {
            probs.truncate(self.top_k);
            // Renormaliza apos Top-K
            let k_sum: f32 = probs.iter().map(|&(_, p)| p).sum();
            if k_sum > 0.0 {
                for item in &mut probs {
                    item.1 /= k_sum;
                }
            }
        }

        // 4. Aplica Top-P (nucleus sampling)
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut cumulative_prob = 0.0f32;
            let mut cutoff_idx = probs.len();
            for (idx, &(_, p)) in probs.iter().enumerate() {
                cumulative_prob += p;
                if cumulative_prob >= self.top_p {
                    cutoff_idx = idx + 1;
                    break;
                }
            }
            probs.truncate(cutoff_idx);
            // Renormaliza
            let p_sum: f32 = probs.iter().map(|&(_, p)| p).sum();
            if p_sum > 0.0 {
                for item in &mut probs {
                    item.1 /= p_sum;
                }
            }
        }

        // 5. Roleta / Amostragem
        let r: f32 = rng.gen();
        let mut cumulative = 0.0f32;
        for (idx, p) in probs {
            cumulative += p;
            if r <= cumulative {
                return idx;
            }
        }

        // Fallback (filtra NaN)
        logits.iter().enumerate()
            .filter(|(_, &v)| v.is_finite())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new(0.8, 40, 0.95)
    }
}
