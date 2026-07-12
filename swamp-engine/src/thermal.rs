// swamp-engine/src/thermal.rs
// ThermalLSC: Coordenador Termico real com base em Telemetria de Frequencia (LSC)
// + RAPL power telemetry + predictive throttle detection (Idea #2)

use std::fs;
use std::time::{Instant, Duration};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Normal,
    Warning,
    Critical,
}

const RAPL_PACKAGE_PATH: &str = "/sys/class/powercap/intel-rapl:0/energy_uj";
const THERMAL_ZONE_PATH: &str = "/sys/class/thermal/thermal_zone0/temp";
const FREQ_PATH: &str = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq";

/// Le a energia acumulada do pacote via RAPL (microjoules).
pub fn read_package_energy_uj() -> u64 {
    fs::read_to_string(RAPL_PACKAGE_PATH)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// Le a temperatura do pacote via ACPI (milli Celsius → Celsius).
pub fn read_package_temp_celsius() -> f32 {
    fs::read_to_string(THERMAL_ZONE_PATH)
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .map(|milli| milli / 1000.0)
        .unwrap_or(45.0)
}

/// Le a frequencia atual do CPU0 via sysfs (kHz).
pub fn read_freq_khz() -> u64 {
    fs::read_to_string(FREQ_PATH)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

pub struct ThermalLSC {
    pub max_observed_freq: u64, // kHz
    pub throttle_count: u32,
    pub warning_temp: f32,
    pub critical_temp: f32,
    pub current_temp: f32,
    pub state: ThermalState,
    last_check: Instant,

    // RAPL power telemetry (Idea #2)
    last_energy_uj: u64,
    last_energy_time: Instant,
    power_draw_uw: u64,          // last measured package power in microwatts
    temp_trend: f32,              // degrees per second (positive = heating)
}

impl ThermalLSC {
    pub fn new(warning_temp: f32, critical_temp: f32) -> Self {
        let now = Instant::now();
        Self {
            max_observed_freq: 0,
            throttle_count: 0,
            warning_temp,
            critical_temp,
            current_temp: 45.0,
            state: ThermalState::Normal,
            last_check: now,
            last_energy_uj: 0,
            last_energy_time: now,
            power_draw_uw: 0,
            temp_trend: 0.0,
        }
    }

    /// Le a frequencia atual do CPU0 via sysfs (kHz)
    pub fn current_freq_khz(&self) -> u64 {
        read_freq_khz()
    }

    /// Atualiza todas as leituras telemétricas (temp, freq, RAPL power).
    /// Deve ser chamada a cada ~10-20 steps de decode (~50-100ms).
    pub fn update(&mut self) {
        let now = Instant::now();

        // Temperatura real
        let new_temp = read_package_temp_celsius();
        let dt = self.last_check.elapsed().as_secs_f32().max(0.001);
        self.temp_trend = (new_temp - self.current_temp) / dt;
        self.current_temp = new_temp;

        // Atualiza estado termico
        self.state = if new_temp >= self.critical_temp {
            ThermalState::Critical
        } else if new_temp >= self.warning_temp {
            ThermalState::Warning
        } else {
            ThermalState::Normal
        };

        // Frequencia
        let freq = read_freq_khz();
        if freq > self.max_observed_freq {
            self.max_observed_freq = freq;
        }

        // RAPL power draw (delta energia / delta tempo)
        let energy = read_package_energy_uj();
        if self.last_energy_uj > 0 {
            let de = energy.saturating_sub(self.last_energy_uj);
            let dt_ns = self.last_energy_time.elapsed().as_nanos().max(1);
            // power (uW) = de (uJ) / dt (s) = de * 1_000_000_000 / dt_ns
            let power = (de as u128).saturating_mul(1_000_000_000) / dt_ns as u128;
            self.power_draw_uw = power.min(200_000_000) as u64;
        }
        self.last_energy_uj = energy;
        self.last_energy_time = now;
        self.last_check = now;
    }

    /// Potência do pacote em microwatts (da última medição RAPL).
    pub fn power_draw_uw(&self) -> u64 {
        self.power_draw_uw
    }

    /// Potência do pacote em watts.
    pub fn power_draw_w(&self) -> f32 {
        self.power_draw_uw as f32 / 1_000_000.0
    }

    /// Retorna true se o throttling é iminente baseado em tendências de
    /// temperatura + potência. Usado pelo scheduler para desviar carga
    /// para a GPU ANTES da frequência cair (Idea #2 — preditivo).
    pub fn predictive_throttle(&self, pl1_limit_uw: u64) -> bool {
        // Critérios de throttling preditivo:
        // 1. Potência > 90% do PL1 → quase batendo no limite térmico
        // 2. Temperatura > warning e subindo (temp_trend > 0)
        // 3. Já estamos em Warning ou Critical
        if self.power_draw_uw > pl1_limit_uw * 90 / 100 && self.temp_trend > 0.0 {
            return true;
        }
        if self.current_temp >= self.warning_temp && self.temp_trend > 1.0 {
            return true; // aquecendo rápido
        }
        if self.state == ThermalState::Critical {
            return true;
        }
        false
    }

    /// Verifica se estamos em throttling baseado na queda de frequencia (AVX-512 power limit hit)
    pub fn is_throttling(&mut self) -> bool {
        if self.last_check.elapsed() < Duration::from_millis(50) {
            return self.throttle_count > 0;
        }

        let freq = read_freq_khz();
        if freq > self.max_observed_freq {
            self.max_observed_freq = freq;
        }

        if self.max_observed_freq > 0 && freq < (self.max_observed_freq * 70 / 100) {
            self.throttle_count = self.throttle_count.saturating_add(1);
            true
        } else {
            self.throttle_count = self.throttle_count.saturating_sub(1);
            false
        }
    }

    /// Calcula a coerência epsilon (Cε) da Teoria LSC (0.0 a 1.0)
    pub fn coherence(&self) -> f64 {
        let freq = read_freq_khz() as f64;
        let max = self.max_observed_freq as f64;
        if max == 0.0 {
            return 1.0;
        }
        (freq / max).clamp(0.0, 1.0)
    }

    pub fn update_temperature(&mut self, temp: f32) -> ThermalState {
        self.current_temp = temp;
        self.state
    }

    pub fn throttle_factor(&self) -> f32 {
        if self.max_observed_freq == 0 {
            return 1.0;
        }
        let c_epsilon = self.coherence();
        if c_epsilon < 0.6 {
            0.3
        } else if c_epsilon < 0.86 {
            0.7
        } else {
            1.0
        }
    }
}

impl Default for ThermalLSC {
    fn default() -> Self {
        Self::new(75.0, 85.0)
    }
}

pub type ThermalCoordinator = ThermalLSC;
