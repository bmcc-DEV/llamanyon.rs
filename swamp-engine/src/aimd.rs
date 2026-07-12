// swamp-engine/src/aimd.rs
// AIMD Resource Ramp — Additive Increase Multiplicative Decrease.
//
// A cada 0.5s sem sinal de estresse (temp, RAPL throttle, latência alta):
//   DOBRA o orçamento de recursos (threads/batch/concorrência).
// No PRIMEIRO sinal ruim:
//   CORTA PELA METADE instantaneamente.
// Sem MSR, sem tocar em frequência — só controle de concorrência.

use std::time::{Duration, Instant};

pub struct AimdRamp {
    budget: f64,
    min_budget: f64,
    max_budget: f64,
    interval: Duration,
    last_assessment: Instant,
    stressed: bool,
    cycles: u64,
    halvings: u64,
}

impl AimdRamp {
    pub fn new() -> Self {
        Self {
            budget: 1.0,
            min_budget: 0.125,
            max_budget: 16.0,
            interval: Duration::from_millis(500),
            last_assessment: Instant::now(),
            stressed: false,
            cycles: 0,
            halvings: 0,
        }
    }

    /// Assess system state. Call once per decode step.
    /// `stressed` = any of: temp > warning, RAPL power > 90% PL1, decode latency spiked.
    /// Returns the current resource budget multiplier.
    pub fn assess(&mut self, stressed: bool) -> f64 {
        let elapsed = self.last_assessment.elapsed();
        if elapsed < self.interval {
            return self.budget;
        }
        self.last_assessment = Instant::now();
        self.cycles += 1;

        if stressed {
            self.budget = (self.budget * 0.5).max(self.min_budget);
            self.halvings += 1;
            self.stressed = true;
        } else {
            self.budget = (self.budget * 2.0).min(self.max_budget);
            self.stressed = false;
        }
        self.budget
    }

    pub fn budget(&self) -> f64 {
        self.budget
    }

    /// Number of active instances = budget rounded up (min 1).
    pub fn concurrency(&self) -> usize {
        (self.budget.ceil() as usize).max(1)
    }

    /// Recommended thread count (base_n_threads * budget, clamped).
    pub fn scaled_threads(&self, base_threads: usize) -> usize {
        let scaled = (base_threads as f64 * self.budget).round() as usize;
        scaled.clamp(1, 12)
    }

    pub fn cycles(&self) -> u64 { self.cycles }
    pub fn halvings(&self) -> u64 { self.halvings }
    pub fn just_stressed(&self) -> bool { self.stressed }
}
