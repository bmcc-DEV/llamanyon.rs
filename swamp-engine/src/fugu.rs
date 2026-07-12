// swamp-engine/src/fugu.rs
// Fugu: orchestrator de estratégias de execução para o LLamañón.rs
//
// Compõe decisões de atenção, GEMV, cache, e especulação baseado no estado atual
// do sistema (tamanho do contexto, pressão de cache, estado térmico, etc.)

use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};

// =========================================================================
// Tipos de estratégia
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionStrategy {
    /// Full quadratic attention sobre todas as posições (default para ctx < 512)
    Full,
    /// Atenção esparsa: janela recente + amostras aleatórias do passado
    Sparse {
        window: usize,
        num_random: usize,
    },
    /// Atenção hierárquica Fugu: janela + sentinel tokens + DSPark cold blocks
    SparseWithDSPark {
        window: usize,
        sentinel_stride: usize,
        num_dspark: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemvStrategy {
    /// Row-partitioned paralelo (atual), com n_threads
    RowParallel(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculationStrategy {
    Disabled,
    /// Número de draft tokens (K)
    Enabled(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    /// Padrão: LRU normal entre Hot/Warm/Cold
    Normal,
    /// Preferir manter páginas em RAM (evitar promoção para VRAM se causar thrash)
    RamPreferred,
}

#[derive(Debug, Clone, Copy)]
pub struct FuguStrategy {
    pub attention: AttentionStrategy,
    pub gemv: GemvStrategy,
    pub speculation: SpeculationStrategy,
    pub cache: CacheStrategy,
}

// =========================================================================
// Métricas de contexto
// =========================================================================

#[derive(Debug, Default, Clone)]
pub struct SystemSnapshot {
    pub seq_len: usize,
    pub cache_pressure: f64,       // 0.0–1.0: fração do cache RAM usada
    pub n_prompt_tokens: usize,
    pub batch_size: usize,
    pub is_prefill: bool,
    pub coherence: f64,            // thermal coherence (c_epsilon)
    pub dspark_accept_rate: f64,   // 0.0–1.0 taxa de aceitação recente
    pub predictive_throttle: bool, // Idea #2: true se throttling é iminente
}

// =========================================================================
// FuguOrchestrator
// =========================================================================

pub struct FuguOrchestrator {
    // Constantes do modelo
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,

    // Limiares configuráveis
    sparse_attention_threshold: usize,    // ativa sparse attention acima deste ctx
    sparse_window: usize,                  // janela recente para sparse attention
    sparse_random_samples: usize,          // amostras aleatórias do passado
    
    // EWMA da taxa de aceitação da DSpark
    accept_rate_ewma: AtomicU64,           // fixed-point Q16.16
    speculation_window: AtomicU32,         // quantos steps para medir
    
    // Estatísticas de estratégia (para diagnóstico)
    n_full_attention: AtomicU64,
    n_sparse_attention: AtomicU64,
}

impl FuguOrchestrator {
    pub fn new(n_layers: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            sparse_attention_threshold: 512,
            sparse_window: 256,
            sparse_random_samples: 32,
            accept_rate_ewma: AtomicU64::new(float_to_fixed(0.5)),
            speculation_window: AtomicU32::new(32),
            n_full_attention: AtomicU64::new(0),
            n_sparse_attention: AtomicU64::new(0),
        }
    }

    /// Decide a estratégia para o próximo passo da inferência.
    pub fn decide(&self, state: &SystemSnapshot) -> FuguStrategy {
        let attention = self.decide_attention(state);
        let gemv = self.decide_gemv(state);
        let speculation = self.decide_speculation(state);
        let cache = self.decide_cache(state);

        FuguStrategy { attention, gemv, speculation, cache }
    }

    fn decide_attention(&self, state: &SystemSnapshot) -> AttentionStrategy {
        if state.is_prefill {
            return AttentionStrategy::Full;
        }

        // Thermal-aware threshold: predictive throttle → sparse earlier to cut power
        let effective_threshold = if state.predictive_throttle {
            self.sparse_attention_threshold.saturating_sub(256)
        } else {
            self.sparse_attention_threshold
        };

        if state.seq_len > effective_threshold {
            self.n_sparse_attention.fetch_add(1, Ordering::Relaxed);
            // Tighter window under throttle to reduce compute
            let window = if state.predictive_throttle {
                self.sparse_window / 2
            } else {
                self.sparse_window
            };
            let dspark_blocks = if state.dspark_accept_rate > 0.0 && state.seq_len > 2048 {
                16
            } else {
                0
            };
            if dspark_blocks > 0 {
                AttentionStrategy::SparseWithDSPark {
                    window,
                    sentinel_stride: 64,
                    num_dspark: dspark_blocks,
                }
            } else {
                AttentionStrategy::Sparse {
                    window,
                    num_random: self.sparse_random_samples,
                }
            }
        } else {
            self.n_full_attention.fetch_add(1, Ordering::Relaxed);
            AttentionStrategy::Full
        }
    }

    fn decide_gemv(&self, _state: &SystemSnapshot) -> GemvStrategy {
        // Por ora, mantém row-parallel com controle via VNpu
        GemvStrategy::RowParallel(0) // 0 = use VNpu default
    }

    fn decide_speculation(&self, state: &SystemSnapshot) -> SpeculationStrategy {
        if state.is_prefill {
            return SpeculationStrategy::Disabled;
        }
        // Só ativa especulação se a taxa de aceitação for razoável
        let rate = self.accept_rate();
        if rate > 0.3 {
            SpeculationStrategy::Enabled(2)
        } else {
            SpeculationStrategy::Disabled
        }
    }

    fn decide_cache(&self, state: &SystemSnapshot) -> CacheStrategy {
        if state.cache_pressure > 0.8 {
            CacheStrategy::RamPreferred
        } else {
            CacheStrategy::Normal
        }
    }

    // =====================================================================
    // Feedback de aceitação da DSpark
    // =====================================================================

    /// Alimenta a taxa de aceitação observada (0.0–1.0).
    pub fn record_acceptance(&self, rate: f64) {
        let old = self.accept_rate_ewma.load(Ordering::Relaxed);
        let decay = 0.3;  // EWMA decay — responde rápido a mudanças
        let new = old as f64 * (1.0 - decay) + float_to_fixed(rate) as f64 * decay;
        self.accept_rate_ewma.store(new as u64, Ordering::Relaxed);
    }

    pub fn accept_rate(&self) -> f64 {
        fixed_to_float(self.accept_rate_ewma.load(Ordering::Relaxed))
    }

    // =====================================================================
    // Diagnóstico
    // =====================================================================

    pub fn diagnostics(&self) -> String {
        let full = self.n_full_attention.load(Ordering::Relaxed);
        let sparse = self.n_sparse_attention.load(Ordering::Relaxed);
        format!(
            "Fugu: full_attn={}, sparse_attn={}, accept_rate={:.1}%",
            full, sparse,
            self.accept_rate() * 100.0,
        )
    }
}

// =========================================================================
// Utilitários fixed-point Q16.16
// =========================================================================

const FP_SHIFT: u64 = 16;
const FP_ONE: u64 = 1 << FP_SHIFT;

fn float_to_fixed(v: f64) -> u64 {
    (v * FP_ONE as f64) as u64
}

fn fixed_to_float(v: u64) -> f64 {
    v as f64 / FP_ONE as f64
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decide_attention_full_short_context() {
        let fugu = FuguOrchestrator::new(22, 32, 4, 64);
        let state = SystemSnapshot {
            seq_len: 100,
            cache_pressure: 0.0,
            n_prompt_tokens: 0,
            batch_size: 1,
            is_prefill: false,
            coherence: 1.0,
            dspark_accept_rate: 0.5,
            predictive_throttle: false,
        };
        let strat = fugu.decide(&state);
        assert_eq!(strat.attention, AttentionStrategy::Full);
    }

    #[test]
    fn test_decide_attention_sparse_long_context() {
        let fugu = FuguOrchestrator::new(22, 32, 4, 64);
        let state = SystemSnapshot {
            seq_len: 1024,
            cache_pressure: 0.0,
            n_prompt_tokens: 0,
            batch_size: 1,
            is_prefill: false,
            coherence: 1.0,
            dspark_accept_rate: 0.5,
            predictive_throttle: false,
        };
        let strat = fugu.decide(&state);
        assert!(matches!(strat.attention, AttentionStrategy::Sparse { .. }));
    }

    #[test]
    fn test_decide_speculation_low_accept_rate() {
        let fugu = FuguOrchestrator::new(22, 32, 4, 64);
        // Record low acceptance multiple times to overcome EWMA inertia
        for _ in 0..10 {
            fugu.record_acceptance(0.05);
        }
        let state = SystemSnapshot {
            seq_len: 100,
            cache_pressure: 0.0,
            n_prompt_tokens: 0,
            batch_size: 1,
            is_prefill: false,
            coherence: 1.0,
            dspark_accept_rate: 0.05,
            predictive_throttle: false,
        };
        let strat = fugu.decide(&state);
        assert_eq!(strat.speculation, SpeculationStrategy::Disabled);
    }

    #[test]
    fn test_decide_speculation_high_accept_rate() {
        let fugu = FuguOrchestrator::new(22, 32, 4, 64);
        for _ in 0..10 {
            fugu.record_acceptance(0.9);
        }
        let state = SystemSnapshot {
            seq_len: 100,
            cache_pressure: 0.0,
            n_prompt_tokens: 0,
            batch_size: 1,
            is_prefill: false,
            coherence: 1.0,
            dspark_accept_rate: 0.9,
            predictive_throttle: false,
        };
        let strat = fugu.decide(&state);
        assert_eq!(strat.speculation, SpeculationStrategy::Enabled(2));
    }

    #[test]
    fn test_fixed_point_roundtrip() {
        let vals = [0.0, 0.5, 1.0, 0.333, 0.99];
        for &v in &vals {
            let fixed = float_to_fixed(v);
            let back = fixed_to_float(fixed);
            assert!((back - v).abs() < 0.001, "roundtrip failed for {}", v);
        }
    }
}
