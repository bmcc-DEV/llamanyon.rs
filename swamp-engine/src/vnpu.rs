use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};

pub const DEFAULT_BUDGET_NS: u64 = 2_000_000;
const EWMA_ALPHA_Q32: u64 = ((1u64 << 32) as f64 * 0.125) as u64;
const EWMA_ONE_MINUS_ALPHA_Q32: u64 = (1u64 << 32) - EWMA_ALPHA_Q32;
const CPU_BACKEND: u8 = 0;
const GPU_BACKEND: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Gpu,
}

pub struct VirtualNpuScheduler {
    ewma_ns_per_row_q32: AtomicU64,
    suggested_threads: AtomicU32,
    budget_ns: u64,
    min_threads: u32,
    max_threads: u32,
    backend: AtomicU8,
}

/// Named set of VirtualNpuSchedulers, one per GEMV operation group in the model.
///
/// Each group has a distinct row-count profile, so independently tuned thread counts
/// converge better than a single shared scheduler would.
pub struct GemvSchedulers {
    pub qkv:    VirtualNpuScheduler,
    pub output: VirtualNpuScheduler,
    pub gateup: VirtualNpuScheduler,
    pub down:   VirtualNpuScheduler,
    pub logits: VirtualNpuScheduler,
}

impl GemvSchedulers {
    pub fn new(budget_ns: u64) -> Self {
        Self {
            qkv:    VirtualNpuScheduler::new(budget_ns),
            output: VirtualNpuScheduler::new(budget_ns),
            gateup: VirtualNpuScheduler::new(budget_ns),
            down:   VirtualNpuScheduler::new(budget_ns),
            logits: VirtualNpuScheduler::new(budget_ns),
        }
    }

    pub fn reset_all(&self) {
        self.qkv.reset();
        self.output.reset();
        self.gateup.reset();
        self.down.reset();
        self.logits.reset();
    }
}

impl VirtualNpuScheduler {
    pub fn new(budget_ns: u64) -> Self {
        Self {
            ewma_ns_per_row_q32: AtomicU64::new(0),
            suggested_threads: AtomicU32::new(4),
            budget_ns,
            min_threads: 1,
            max_threads: 6,
            backend: AtomicU8::new(CPU_BACKEND),
        }
    }

    pub fn schedule(&self) -> (usize, Backend) {
        let nt = self.suggested_threads.load(Ordering::Relaxed);
        let be = match self.backend.load(Ordering::Relaxed) {
            GPU_BACKEND => Backend::Gpu,
            _ => Backend::Cpu,
        };
        (nt.clamp(self.min_threads, self.max_threads) as usize, be)
    }

    pub fn record(&self, elapsed_ns: u64, n_rows: usize) {
        if n_rows == 0 {
            return;
        }
        let ns_per_row = elapsed_ns / n_rows as u64;

        let sample_q32 = (ns_per_row as u128) << 32;
        loop {
            let old = self.ewma_ns_per_row_q32.load(Ordering::Relaxed);
            let new = if old == 0 {
                sample_q32 as u64
            } else {
                let ew = (sample_q32 * EWMA_ALPHA_Q32 as u128
                    + (old as u128) * EWMA_ONE_MINUS_ALPHA_Q32 as u128)
                    >> 32;
                ew as u64
            };
            if self
                .ewma_ns_per_row_q32
                .compare_exchange_weak(old, new, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        let ewma_q32 = self.ewma_ns_per_row_q32.load(Ordering::Relaxed);
        let expected_total = (ewma_q32 as u128 * n_rows as u128) >> 32;
        let current_nt = self.suggested_threads.load(Ordering::Relaxed);

        let new_nt = if expected_total > self.budget_ns as u128 && current_nt < self.max_threads {
            current_nt + 1
        } else if expected_total < (self.budget_ns as u128) / 2 && current_nt > self.min_threads {
            current_nt - 1
        } else {
            current_nt
        };

        self.suggested_threads.store(new_nt, Ordering::Release);
    }

    #[allow(dead_code)]
    pub fn set_backend(&self, backend: Backend) {
        self.backend.store(
            match backend {
                Backend::Cpu => CPU_BACKEND,
                Backend::Gpu => GPU_BACKEND,
            },
            Ordering::Release,
        );
    }

    pub fn current_estimate_ns(&self) -> u64 {
        let q32 = self.ewma_ns_per_row_q32.load(Ordering::Relaxed);
        q32 >> 32
    }

    pub fn reset(&self) {
        self.ewma_ns_per_row_q32.store(0, Ordering::Release);
        self.suggested_threads.store(4, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vnpu_schedule_default() {
        let vnpu = VirtualNpuScheduler::new(DEFAULT_BUDGET_NS);
        let (nt, be) = vnpu.schedule();
        assert_eq!(nt, 4);
        assert_eq!(be, Backend::Cpu);
    }

    #[test]
    fn test_vnpu_record_under_budget() {
        let vnpu = VirtualNpuScheduler::new(DEFAULT_BUDGET_NS);
        // 1000 rows completed in 500μs → 500ns/row → well under 2ms budget
        vnpu.record(500_000, 1000);
        let (nt, _) = vnpu.schedule();
        // Should reduce threads (half budget headroom)
        assert_eq!(nt, 3);
    }

    #[test]
    fn test_vnpu_record_over_budget() {
        let vnpu = VirtualNpuScheduler::new(DEFAULT_BUDGET_NS);
        // 100 rows completed in 3ms → 30000ns/row → over 2ms budget
        vnpu.record(3_000_000, 100);
        let (nt, _) = vnpu.schedule();
        // Should increase threads
        assert_eq!(nt, 5);
    }

    #[test]
    fn test_vnpu_backend_default_cpu() {
        let vnpu = VirtualNpuScheduler::new(DEFAULT_BUDGET_NS);
        assert_eq!(vnpu.schedule().1, Backend::Cpu);
    }

    #[test]
    fn test_vnpu_backend_gpu_reserved() {
        let vnpu = VirtualNpuScheduler::new(DEFAULT_BUDGET_NS);
        vnpu.set_backend(Backend::Gpu);
        assert_eq!(vnpu.schedule().1, Backend::Gpu);
    }

    #[test]
    fn test_vnpu_reset() {
        let vnpu = VirtualNpuScheduler::new(DEFAULT_BUDGET_NS);
        vnpu.record(3_000_000, 100);
        assert_eq!(vnpu.schedule().0, 5);
        vnpu.reset();
        assert_eq!(vnpu.schedule().0, 4);
    }
}
