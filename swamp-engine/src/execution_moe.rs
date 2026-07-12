use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::OnceLock;

/// If only one expert is available, no routing needed.
pub fn num_available_experts() -> usize {
    available_experts().len()
}

/// Global singleton router shared across all GEMV calls.
pub fn global_router() -> &'static ExpertRouter {
    static ROUTER: OnceLock<ExpertRouter> = OnceLock::new();
    // Explore frequently during warmup (interval=64), then less often
    ROUTER.get_or_init(|| ExpertRouter::new(128))
}

// =========================================================================
// GemmExpert — execution backends ("experts")
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmExpert {
    Vnni  = 0,
    Avx2  = 1,
    Scalar = 2,
}

impl GemmExpert {
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => GemmExpert::Vnni,
            1 => GemmExpert::Avx2,
            _ => GemmExpert::Scalar,
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            GemmExpert::Vnni => "vnni",
            GemmExpert::Avx2 => "avx2",
            GemmExpert::Scalar => "scalar",
        }
    }
}

// =========================================================================
// OpProfile — operation class for generalization
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpProfile {
    pub row_bucket: u32,
    pub col_bucket: u32,
    pub is_batched: bool,
}

fn bucket_dim(v: usize) -> u32 {
    if v <= 256 { 0 }
    else if v <= 1024 { 1 }
    else if v <= 4096 { 2 }
    else if v <= 16384 { 3 }
    else { 4 }
}

impl OpProfile {
    pub fn new(n_rows: usize, n_cols: usize, is_batched: bool) -> Self {
        Self {
            row_bucket: bucket_dim(n_rows),
            col_bucket: bucket_dim(n_cols),
            is_batched,
        }
    }
}

// =========================================================================
// ExpertRouter — MoE routing table
// =========================================================================

const NUM_EXPERTS: usize = 3;
const EWMA_ALPHA_Q32: u64 = ((1u64 << 32) as f64 * 0.125) as u64;
const ONE_MINUS_ALPHA_Q32: u64 = (1u64 << 32) - EWMA_ALPHA_Q32;

pub struct ExpertRouter {
    /// EWMA ns-per-call per (profile_idx * NUM_EXPERTS + expert_idx)
    costs: Vec<AtomicU64>,
    /// Sample count per slot
    samples: Vec<AtomicU32>,
    /// Exploration: try random expert every N calls
    explore_interval: u32,
    call_count: AtomicU32,
}

impl ExpertRouter {
    pub fn new(explore_interval: u32) -> Self {
        Self {
            costs: (0..64 * NUM_EXPERTS).map(|_| AtomicU64::new(0)).collect(),
            samples: (0..64 * NUM_EXPERTS).map(|_| AtomicU32::new(0)).collect(),
            explore_interval,
            call_count: AtomicU32::new(0),
        }
    }

    fn slot(&self, profile: &OpProfile, expert: GemmExpert) -> usize {
        let pi = (profile.row_bucket * 5 + profile.col_bucket) * 2 + profile.is_batched as u32;
        pi as usize * NUM_EXPERTS + expert.index()
    }

    /// Select the best expert for this operation profile.
    /// Uses epsilon-greedy: explore random expert every `explore_interval` calls.
    pub fn select(&self, profile: &OpProfile) -> GemmExpert {
        let avail = available_experts();
        if avail.len() <= 1 {
            return avail[0];
        }

        let cc = self.call_count.fetch_add(1, Ordering::Relaxed);
        if self.explore_interval > 0 && cc % self.explore_interval == 0 {
            // Epsilon-greedy: pick random available expert
            let i = (cc.wrapping_mul(2654435761) >> 16) as usize % avail.len();
            return avail[i];
        }

        // Greedy: pick cheapest (lowest EWMA) among available with >0 samples
        let mut best = avail[0];
        let mut best_cost = u64::MAX;
        for &exp in &avail {
            let s = self.slot(profile, exp);
            let c = self.costs[s].load(Ordering::Relaxed);
            let n = self.samples[s].load(Ordering::Relaxed);
            if n > 0 && c < best_cost {
                best_cost = c;
                best = exp;
            }
        }
        best
    }

    /// Record elapsed_ns for a completed operation.
    pub fn record(&self, profile: &OpProfile, expert: GemmExpert, elapsed_ns: u64) {
        let s = self.slot(profile, expert);
        self.samples[s].fetch_add(1, Ordering::Relaxed);

        loop {
            let old = self.costs[s].load(Ordering::Relaxed);
            let new = if old == 0 {
                elapsed_ns
            } else {
                (((elapsed_ns as u128) * EWMA_ALPHA_Q32 as u128
                    + (old as u128) * ONE_MINUS_ALPHA_Q32 as u128) >> 32) as u64
            };
            if self.costs[s]
                .compare_exchange_weak(old, new, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Report: show cost per expert for a profile
    pub fn report_profile(&self, profile: &OpProfile) -> Vec<(GemmExpert, u64, u32)> {
        let avail = available_experts();
        avail.iter().map(|&exp| {
            let s = self.slot(profile, exp);
            (exp, self.costs[s].load(Ordering::Relaxed), self.samples[s].load(Ordering::Relaxed))
        }).collect()
    }
}

/// List of HW-available experts (checked once at init).
pub fn available_experts() -> Vec<GemmExpert> {
    let mut v = Vec::with_capacity(NUM_EXPERTS);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vnni") {
            v.push(GemmExpert::Vnni);
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            v.push(GemmExpert::Avx2);
        }
    }
    v.push(GemmExpert::Scalar);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_defaults_to_first_available() {
        let router = ExpertRouter::new(0);
        let profile = OpProfile::new(2048, 2048, false);
        let exp = router.select(&profile);
        let avail = available_experts();
        assert!(avail.contains(&exp));
    }

    #[test]
    fn test_record_and_select() {
        let router = ExpertRouter::new(0);
        let profile = OpProfile::new(2048, 2048, false);
        // Record Vnni as very slow
        router.record(&profile, GemmExpert::Vnni, 1_000_000);
        // Record Avx2 as fast
        router.record(&profile, GemmExpert::Avx2, 100_000);
        // Should pick Avx2
        let exp = router.select(&profile);
        assert_eq!(exp, GemmExpert::Avx2);
    }

    #[test]
    fn test_bucket_dim() {
        assert_eq!(bucket_dim(256), 0);
        assert_eq!(bucket_dim(512), 1);
        assert_eq!(bucket_dim(2048), 2);
        assert_eq!(bucket_dim(8192), 3);
        assert_eq!(bucket_dim(32000), 4);
    }
}
