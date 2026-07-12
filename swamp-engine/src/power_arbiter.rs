// swamp-engine/src/power_arbiter.rs
// PowerArbiter — arbitragem de potência compartilhada CPU + iGPU.
//
// Diferente do dGPU (que tem rail de energia próprio — GTX 1650 ~75W PCIe),
// a iGPU Intel compete pelo MESMO orçamento de potência do pacote (PL1/PL2)
// que os núcleos AVX-512 da CPU. Rodar GEMV no talo E iGPU no talo ao
// mesmo tempo pode fazer ambos brigarem pelo mesmo budget térmico.
//
// O PowerArbiter lê RAPL + temperatura + flag de throttle e decide como
// dividir o orçamento AIMD entre threads CPU e frequência/voltagem iGPU.

use crate::thermal::{read_package_energy_uj, read_package_temp_celsius};
use std::time::{Duration, Instant};

/// Fraction of the AIMD budget that goes to CPU threads (rest → iGPU).
pub struct PowerArbiter {
    cpu_fraction: f64,      // 0.0 = all to iGPU, 1.0 = all to CPU
    igpu_fraction: f64,     // derived: 1.0 - cpu_fraction
    last_assessment: Instant,
    interval: Duration,
    // telemetry snapshot
    last_energy: u64,
    last_time: Instant,
    power_w: f64,
    temp_c: f64,
    temp_trend: f64,
}

impl PowerArbiter {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            cpu_fraction: 0.7,     // start 70/30 favoring CPU
            igpu_fraction: 0.3,
            last_assessment: now,
            interval: Duration::from_millis(500),
            last_energy: 0,
            last_time: now,
            power_w: 0.0,
            temp_c: 45.0,
            temp_trend: 0.0,
        }
    }

    /// Read fresh telemetry and reassess the CPU/iGPU split.
    /// Call every ~500ms (same cadence as AIMD).
    pub fn reassess(&mut self, pl1_watts: f64) -> (f64, f64) {
        let now = Instant::now();
        let dt = self.last_assessment.elapsed().as_secs_f64().max(0.001);
        if dt < self.interval.as_secs_f64() {
            return (self.cpu_fraction, self.igpu_fraction);
        }

        // Fresh telemetry
        let energy = read_package_energy_uj();
        if self.last_energy > 0 {
            let de = energy.saturating_sub(self.last_energy);
            let dt_s = self.last_time.elapsed().as_secs_f64().max(0.001);
            self.power_w = de as f64 / 1_000_000.0 / dt_s;
        }
        self.last_energy = energy;
        self.last_time = now;

        let new_temp = read_package_temp_celsius() as f64;
        self.temp_trend = (new_temp - self.temp_c) / dt;
        self.temp_c = new_temp;

        // Decision logic:
        // If CPU power is high AND temp rising → shift to iGPU (more efficient per FLOP)
        // If total power < 70% PL1 → shift to CPU (AVX-512 is faster)
        // If thermals healthy → favor CPU

        if self.power_w > pl1_watts * 0.9 && self.temp_trend > 0.5 {
            // Near throttle: shift aggressively to iGPU
            self.cpu_fraction = (self.cpu_fraction - 0.1).max(0.2);
        } else if self.temp_c > 80.0 && self.temp_trend > 0.0 {
            // Hot and rising: gradual shift to iGPU
            self.cpu_fraction = (self.cpu_fraction - 0.05).max(0.3);
        } else if self.power_w < pl1_watts * 0.6 && self.temp_c < 70.0 {
            // Cool and low power: favor CPU
            self.cpu_fraction = (self.cpu_fraction + 0.05).min(0.85);
        }
        // else: maintain current split

        self.igpu_fraction = 1.0 - self.cpu_fraction;
        self.last_assessment = now;
        (self.cpu_fraction, self.igpu_fraction)
    }

    /// How many of `total_threads` should go to CPU GEMV work.
    pub fn cpu_threads(&self, total_threads: usize) -> usize {
        (total_threads as f64 * self.cpu_fraction).round().max(1.0) as usize
    }

    /// How many staging slices should iGPU handle.
    pub fn igpu_slices(&self, total_slices: usize) -> usize {
        (total_slices as f64 * self.igpu_fraction).round().max(0.0) as usize
    }

    pub fn cpu_fraction(&self) -> f64 { self.cpu_fraction }
    pub fn igpu_fraction(&self) -> f64 { self.igpu_fraction }
    pub fn power_w(&self) -> f64 { self.power_w }
    pub fn temp_c(&self) -> f64 { self.temp_c }
}
