// swamp-engine/src/timewarp.rs
// TimeWarp: clock virtual com multiplicador variável para controle de intensidade.
//
// Ciclo:
//   multiplier = 0.5 → aumenta gradualmente até 3.0x
//   Quando atinge overload_threshold, snapback para 0.5x
//   step_size acelera a cada ciclo (evolução)

const INITIAL_MULTIPLIER: f64 = 0.5;
const MAX_MULTIPLIER: f64 = 3.0;
const SNAPBACK_TARGET: f64 = 0.5;
const STEP_BASE: f64 = 0.005; // increase per tick

pub struct TimeWarp {
    multiplier: f64,
    step: f64,
    cycle: u64,
    tick: u64,
    overload_threshold: f64,
}

impl TimeWarp {
    pub fn new() -> Self {
        Self {
            multiplier: INITIAL_MULTIPLIER,
            step: STEP_BASE,
            cycle: 0,
            tick: 0,
            overload_threshold: MAX_MULTIPLIER,
        }
    }

    /// Advance one tick. Returns the current virtual multiplier.
    /// On snapback, returns the NEGATIVE of the pre-snap multiplier.
    pub fn tick(&mut self) -> f64 {
        self.tick += 1;
        self.multiplier += self.step;
        if self.multiplier >= self.overload_threshold {
            let pre_snap = self.multiplier;
            self.multiplier = SNAPBACK_TARGET;
            self.cycle += 1;
            self.step *= 1.10;
            self.overload_threshold = (MAX_MULTIPLIER - 0.2 * self.cycle as f64).max(1.5);
            return -pre_snap;
        }
        self.multiplier
    }

    /// True if we just snapped back in this tick.
    pub fn did_snapback(result: f64) -> bool {
        result < 0.0
    }

    /// Current multiplier (0.5x to 3.0x+)
    pub fn current(&self) -> f64 {
        self.multiplier
    }

    /// True if we're in the "slow zone" where aggressive prefetch is beneficial
    pub fn prefetch_aggressive(&self) -> bool {
        self.multiplier < 1.5
    }

    /// True if we just snapped back (call after tick() returns negative)
    pub fn just_snapped_back(&self) -> bool {
        false // checked via tick() return value
    }

    /// Cycle count (how many snapbacks occurred)
    pub fn cycle(&self) -> u64 {
        self.cycle
    }
}
