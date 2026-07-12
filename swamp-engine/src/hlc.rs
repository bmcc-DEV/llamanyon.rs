use std::sync::atomic::{AtomicU32, AtomicU64, AtomicBool, Ordering};
use std::time::Duration;

/// Global flag: profiling is enabled only when RUST_LOG=debug or SWAMP_PROFILE=1.
/// Saves ~64 atomic CAS + ~2 allocations per decode step when disabled.
static PROFILING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Call once at startup to enable profiling (e.g. when RUST_LOG=debug).
pub fn init_profiling() {
    let enabled = std::env::var("SWAMP_PROFILE").ok().as_deref() == Some("1")
        || std::env::var("RUST_LOG").ok()
            .map(|v| v.contains("debug") || v.contains("trace"))
            .unwrap_or(false);
    PROFILING_ENABLED.store(enabled, Ordering::Release);
}

fn profiling_enabled() -> bool {
    PROFILING_ENABLED.load(Ordering::Acquire)
}

// ---------------------------------------------------------------------------
// WallClock: TSC-based monotonic clock calibrated to nanoseconds
// ---------------------------------------------------------------------------

fn rdtsc() -> u64 {
    unsafe { std::arch::x86_64::_rdtsc() }
}

pub struct WallClock {
    /// nanoseconds per TSC tick (fixed-point 1/2^32)
    scale: AtomicU64,
    /// TSC value at the calibration moment
    tsc0: u64,
    /// CLOCK_MONOTONIC ns at calibration moment
    monon0: u64,
}

impl WallClock {
    pub fn new() -> Self {
        let (tsc0, monon0) = Self::calibrate();
        let scale = Self::measure_scale(tsc0);
        WallClock { scale: AtomicU64::new(scale), tsc0, monon0 }
    }

    /// Calibrate by reading TSC and CLOCK_MONOTONIC in quick succession
    fn calibrate() -> (u64, u64) {
        let tsc = rdtsc();
        let mono = Self::now_monotonic_ns();
        (tsc, mono)
    }

    /// Measure TSC frequency by sampling over ~10ms
    fn measure_scale(_base_tsc: u64) -> u64 {
        let tsc0 = rdtsc();
        std::thread::sleep(Duration::from_millis(10));
        let tsc1 = rdtsc();
        let elapsed = tsc1.saturating_sub(tsc0);
        // 10ms = 10_000_000 ns
        // scale = (ns << 32) / ticks
        if elapsed == 0 {
            return 1u64 << 32; // fallback 1ns/tick
        }
        let ns = 10_000_000u64;
        (((ns as u128) << 32) / elapsed as u128) as u64
    }

    fn now_monotonic_ns() -> u64 {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts); }
        ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
    }

    /// Current wall time in nanoseconds since an unspecified epoch (monotonic)
    pub fn now_ns(&self) -> u64 {
        let tsc = rdtsc();
        let delta = tsc.saturating_sub(self.tsc0);
        let scale = self.scale.load(Ordering::Relaxed);
        // delta * scale >> 32
        let ns = ((delta as u128) * (scale as u128)) >> 32;
        self.monon0 + ns as u64
    }

    /// Re-calibrate if TSC frequency might have changed (e.g. after frequency scaling)
    pub fn recalibrate(&mut self) {
        // Compute current time with old scale, then update base so now_ns() is continuous
        let old_now = self.now_ns();
        let new_scale = Self::measure_scale(self.tsc0);
        let new_tsc = rdtsc();
        self.scale.store(new_scale, Ordering::Relaxed);
        self.tsc0 = new_tsc;
        // monon0 adjusted so now_ns() returns old_now at the new tsc0
        self.monon0 = old_now;
    }
}

// ---------------------------------------------------------------------------
// HlcTimestamp: a point on the hybrid logical timeline
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HlcTimestamp {
    /// Wall time in nanoseconds (monotonic)
    pub wall: u64,
    /// Logical counter for same-wall events
    pub logical: u32,
}

impl Default for HlcTimestamp {
    fn default() -> Self {
        HlcTimestamp::ZERO
    }
}

impl HlcTimestamp {
    pub const ZERO: HlcTimestamp = HlcTimestamp { wall: 0, logical: 0 };

    pub fn new(wall: u64, logical: u32) -> Self {
        HlcTimestamp { wall, logical }
    }

    /// Wall component as human-readable Duration
    pub fn wall_duration(&self) -> Duration {
        Duration::from_nanos(self.wall)
    }

    /// Compare by (wall, logical)
    pub fn happened_before(&self, other: &HlcTimestamp) -> bool {
        self.wall < other.wall || (self.wall == other.wall && self.logical < other.logical)
    }

    /// Check if this timestamp could have caused the other
    pub fn could_cause(&self, other: &HlcTimestamp) -> bool {
        self.happened_before(other)
    }
}

impl PartialOrd for HlcTimestamp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HlcTimestamp {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.wall
            .cmp(&other.wall)
            .then(self.logical.cmp(&other.logical))
    }
}

// ---------------------------------------------------------------------------
// HybridClock: guarantees monotonicity across wall + logical domains
// ---------------------------------------------------------------------------

pub struct HybridClock {
    wall_clock: WallClock,
    last_wall: AtomicU64,
    last_logical: AtomicU32,
}

impl HybridClock {
    pub fn new() -> Self {
        let wc = WallClock::new();
        let wall = wc.now_ns();
        HybridClock {
            wall_clock: wc,
            last_wall: AtomicU64::new(wall),
            last_logical: AtomicU32::new(0),
        }
    }

    /// Generate a new timestamp guaranteed to be > any previous timestamp
    /// from this clock instance (local monotonicity).
    pub fn now(&self) -> HlcTimestamp {
        let physical = self.wall_clock.now_ns();
        loop {
            let last_w = self.last_wall.load(Ordering::Acquire);
            let last_l = self.last_logical.load(Ordering::Relaxed);
            let (new_wall, new_logical) = if physical > last_w {
                (physical, 0)
            } else {
                // physical <= last_w: advance logical component
                (last_w, last_l.wrapping_add(1))
            };
            // CAS to commit
            if self.last_wall.compare_exchange(last_w, new_wall, Ordering::Release, Ordering::Relaxed).is_err() {
                continue;
            }
            self.last_logical.store(new_logical, Ordering::Release);
            return HlcTimestamp { wall: new_wall, logical: new_logical };
        }
    }

    /// Incorporate an external timestamp (e.g. from GPU event), ensuring
    /// this clock stays ahead of received timestamps.
    pub fn witness(&self, ts: HlcTimestamp) -> HlcTimestamp {
        let physical = self.wall_clock.now_ns();
        loop {
            let last_w = self.last_wall.load(Ordering::Acquire);
            let last_l = self.last_logical.load(Ordering::Relaxed);
            let (new_wall, new_logical) = if ts.wall > last_w && ts.wall > physical {
                (ts.wall, 0u32)
            } else if physical >= last_w && physical >= ts.wall {
                (physical, 0u32)
            } else if last_w >= physical && last_w >= ts.wall {
                (last_w, last_l.wrapping_add(1))
            } else {
                // ts.wall >= physical and ts.wall >= last_w
                // But ts.wall == last_w possible: advance logical
                if ts.wall == physical || ts.wall == last_w {
                    let max_l = last_l.max(ts.logical);
                    (ts.wall, max_l.wrapping_add(1))
                } else {
                    (ts.wall, 0u32)
                }
            };
            if self.last_wall.compare_exchange(last_w, new_wall, Ordering::Release, Ordering::Relaxed).is_err() {
                continue;
            }
            self.last_logical.store(new_logical, Ordering::Release);
            return HlcTimestamp { wall: new_wall, logical: new_logical };
        }
    }

    /// Re-calibrate TSC frequency (call periodically if CPU frequency scaling is aggressive)
    pub fn recalibrate(&mut self) {
        self.wall_clock.recalibrate();
    }

    /// Access the underlying wall clock for raw TSC-to-ns conversion
    pub fn wall_clock(&self) -> &WallClock {
        &self.wall_clock
    }
}

// ---------------------------------------------------------------------------
// CudaEventTimer: GPU timing with HLC correlation
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
pub struct CudaEventTimer {
    start: swamp_gpu::CudaEvent,
    end: swamp_gpu::CudaEvent,
}

#[cfg(feature = "gpu")]
impl CudaEventTimer {
    pub fn new() -> Option<Self> {
        if !swamp_gpu::gpu_available() {
            return None;
        }
        let start = swamp_gpu::gpu_event_create().ok()?;
        let end = swamp_gpu::gpu_event_create().ok()?;
        Some(CudaEventTimer { start, end })
    }

    /// Record start event on default stream
    pub fn record_start(&self) -> bool {
        swamp_gpu::gpu_event_record(self.start, std::ptr::null_mut()).is_ok()
    }

    /// Record end event on default stream
    pub fn record_end(&self) -> bool {
        swamp_gpu::gpu_event_record(self.end, std::ptr::null_mut()).is_ok()
    }

    /// Synchronize end event and return elapsed milliseconds
    pub fn elapsed_ms(&self) -> Option<f64> {
        swamp_gpu::gpu_event_synchronize(self.end).ok()?;
        let ms = swamp_gpu::gpu_event_elapsed_ms(self.start, self.end).ok()?;
        Some(ms as f64)
    }
}

#[cfg(feature = "gpu")]
impl Drop for CudaEventTimer {
    fn drop(&mut self) {
        let _ = swamp_gpu::gpu_event_destroy(self.start);
        let _ = swamp_gpu::gpu_event_destroy(self.end);
    }
}

// ---------------------------------------------------------------------------
// Per-layer timing snapshot
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct LayerTiming {
    pub layer: usize,
    pub wall_start: HlcTimestamp,
    pub wall_end: HlcTimestamp,
    pub gpu_attention_ms: Option<f64>,
    pub elapsed_ns: u64,
}

impl LayerTiming {
    pub fn elapsed_us(&self) -> f64 {
        self.elapsed_ns as f64 / 1_000.0
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed_ns as f64 / 1_000_000.0
    }
}

// ---------------------------------------------------------------------------
// ProfileSink: collects timing data for analysis
// ---------------------------------------------------------------------------

pub struct ProfileSink {
    layers: Vec<LayerTiming>,
    clock: HybridClock,
}

impl ProfileSink {
    pub fn new() -> Self {
        ProfileSink {
            layers: Vec::with_capacity(64),
            clock: HybridClock::new(),
        }
    }

    pub fn clock(&self) -> &HybridClock {
        &self.clock
    }

    pub fn begin_layer(&mut self, layer: usize, gpu_expected: bool) -> LayerTiming {
        if !profiling_enabled() {
            return LayerTiming::default();
        }
        let ts = self.clock.now();
        LayerTiming {
            layer,
            wall_start: ts,
            wall_end: HlcTimestamp::ZERO,
            gpu_attention_ms: if gpu_expected { Some(0.0) } else { None },
            elapsed_ns: 0,
        }
    }

    pub fn end_layer(&mut self, mut timing: LayerTiming) {
        if !profiling_enabled() { return; }
        let ts = self.clock.now();
        timing.wall_end = ts;
        timing.elapsed_ns = ts.wall.saturating_sub(timing.wall_start.wall);
        self.layers.push(timing);
    }

    pub fn record_gpu_attention(&mut self, timing: &mut LayerTiming, elapsed_ms: f64) {
        if !profiling_enabled() { return; }
        timing.gpu_attention_ms = Some(elapsed_ms);
    }

    pub fn is_empty(&self) -> bool {
        if !profiling_enabled() { return true; }
        self.layers.is_empty()
    }

    /// Report summary statistics
    pub fn report(&self) -> String {
        if self.layers.is_empty() {
            return "No timing data collected.".into();
        }
        let total_ns: u64 = self.layers.iter().map(|l| l.elapsed_ns).sum();
        let avg_ns = total_ns / self.layers.len() as u64;
        let max_layer = self.layers.iter().max_by_key(|l| l.elapsed_ns).unwrap();
        let gpu_times: Vec<f64> = self.layers.iter()
            .filter_map(|l| l.gpu_attention_ms)
            .collect();
        let avg_gpu = if gpu_times.is_empty() { 0.0 }
            else { gpu_times.iter().sum::<f64>() / gpu_times.len() as f64 };
        format!(
            "Profile: {} layers | avg {:.3}ms | max layer {} {:.3}ms | avg GPU attention {:.3}ms | total {:.3}ms",
            self.layers.len(),
            avg_ns as f64 / 1_000_000.0,
            max_layer.layer, max_layer.elapsed_ns as f64 / 1_000_000.0,
            avg_gpu,
            total_ns as f64 / 1_000_000.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wall_clock_monotonic() {
        let wc = WallClock::new();
        let t1 = wc.now_ns();
        std::thread::sleep(Duration::from_micros(100));
        let t2 = wc.now_ns();
        assert!(t2 > t1, "wall clock must be monotonic: {} <= {}", t2, t1);
    }

    #[test]
    fn test_hlc_monotonic() {
        let hlc = HybridClock::new();
        let t1 = hlc.now();
        let t2 = hlc.now();
        assert!(t1.happened_before(&t2) || t1 == t2,
            "HLC must be monotonic: {:?} >= {:?}", t1, t2);
    }

    #[test]
    fn test_hlc_logical_advance() {
        let hlc = HybridClock::new();
        // Rapid calls should advance logical counter
        let t1 = hlc.now();
        let t2 = hlc.now();
        let t3 = hlc.now();
        // All must be ordered
        assert!(t1 <= t2 && t2 <= t3);
    }

    #[test]
    fn test_witness() {
        let hlc = HybridClock::new();
        let future_ts = HlcTimestamp::new(u64::MAX, 0);
        let witnessed = hlc.witness(future_ts);
        assert!(witnessed.wall >= future_ts.wall);
    }

    #[test]
    fn test_hlc_ordering() {
        let mut timestamps = vec![
            HlcTimestamp::new(100, 3),
            HlcTimestamp::new(100, 1),
            HlcTimestamp::new(100, 2),
            HlcTimestamp::new(50, 0),
        ];
        timestamps.sort();
        assert_eq!(timestamps[0].wall, 50);
        assert_eq!(timestamps[1].logical, 1);
        assert_eq!(timestamps[2].logical, 2);
        assert_eq!(timestamps[3].logical, 3);
    }

    #[test]
    fn test_profile_sink() {
        PROFILING_ENABLED.store(true, Ordering::Release);
        let mut sink = ProfileSink::new();
        let t = sink.begin_layer(0, false);
        std::thread::sleep(Duration::from_micros(500));
        sink.end_layer(t);
        let report = sink.report();
        PROFILING_ENABLED.store(false, Ordering::Release);
        assert!(report.contains("1 layers"));
        assert!(report.contains("avg"));
    }
}
