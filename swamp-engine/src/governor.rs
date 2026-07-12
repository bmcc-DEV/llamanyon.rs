// swamp-engine/src/governor.rs
// ResourceGovernor — unifica AimdRamp, PowerArbiter, ModelRegistry, ModelSwapper,
// ThermalLSC numa camada única que decide, a cada ciclo:
//   - quem roda (admissão por memória + headroom)
//   - com qual flag (prefill_chunk_size, n_threads)
//   - em qual tier (VRAM/RAM/CPU, via swapper)

use crate::aimd::AimdRamp;
use crate::power_arbiter::PowerArbiter;
use crate::thermal::{ThermalLSC, read_package_temp_celsius, read_freq_khz};
use crate::model_registry::{ModelRegistry, MemoryTier, ModelEntry};
use crate::model_swapper::{ModelSwapper, SwapAction};
use std::time::{Duration, Instant};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Workload classification — observa cadência real, não adivinha intenção
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkloadClass {
    /// Usuário esperando: gaps curtos, prompts pequenos → prioriza latência
    Interactive { avg_gap_ms: f64 },
    /// Fila acumulando → prioriza throughput agregado
    Batch { queue_depth: usize },
    /// Modelo pequeno de apoio (classificador, DSPark, roteador) — CPU-only,
    /// footprint desprezível, roda nos threads ociosos sem passar pelo SwampVM
    Cognitive,
}

/// Estado observado de uma sessão para classificação
#[derive(Debug, Clone)]
pub struct SessionTelemetry {
    pub gap_from_last_ms: f64,
    pub prompt_len: usize,
    pub total_processed: u64,
}

/// Sessão registrada no Governor
#[derive(Debug)]
pub struct SessionRecord {
    pub id: SessionId,
    pub class: WorkloadClass,
    pub model_name: String,
    pub generation: u64,         // incrementa a cada cycle, usado pra stale detection
    pub last_gap_ms: f64,
    pub total_tokens: u64,
    pub created_at: Instant,
    pub last_active: Instant,
}

pub type SessionId = u64;

// ---------------------------------------------------------------------------
// ResourceGovernor
// ---------------------------------------------------------------------------

pub struct ResourceGovernor {
    // Peças existentes — mantidas intactas
    pub aimd: AimdRamp,
    pub power: PowerArbiter,
    pub thermal: ThermalLSC,
    pub registry: ModelRegistry,
    pub swapper: ModelSwapper,
    pub max_n_threads: usize,

    // Sessões ativas
    sessions: Vec<SessionRecord>,
    next_session_id: SessionId,

    // Estado derivado (recalculado a cada cycle)
    pub prefill_chunk_size: usize,
    pub cpu_threads: usize,
    pub gpu_model_threshold: usize,  // bytes: modelos >= este vão pra GPU
    pub total_budget: f64,           // aimd.budget() + ajuste térmico

    // Intervalo de reavaliação
    interval: Duration,
    last_cycle: Instant,
    cycle_count: u64,

    // Constantes
    pub min_safe_headroom: f64,      // fração do budget que não pode ser consumida
}

impl ResourceGovernor {
    pub fn new(
        vram_bytes: usize,      // ex: GTX 1650 = 4_294_967_296
        max_n_threads: usize,
        pl1_watts: f64,
    ) -> Self {
        let now = Instant::now();
        Self {
            aimd: AimdRamp::new(),
            power: PowerArbiter::new(),
            thermal: ThermalLSC::default(),
            registry: ModelRegistry::new(),
            swapper: ModelSwapper::new(vram_bytes),
            max_n_threads,

            sessions: Vec::new(),
            next_session_id: 1,

            prefill_chunk_size: 4096,
            cpu_threads: max_n_threads,
            gpu_model_threshold: 3_000_000_000,  // ~3GB → só modelos grandes vão pra GPU
            total_budget: 1.0,

            interval: Duration::from_millis(500), // mesma cadência do AIMD
            last_cycle: now,
            cycle_count: 0,

            min_safe_headroom: 0.15, // 15% do budget reservado pra rajadas
        }
    }

    // =====================================================================
    // Session management
    // =====================================================================

    pub fn register_session(
        &mut self,
        model_name: &str,
        model_size_bytes: usize,
        telemetry: SessionTelemetry,
    ) -> Option<SessionId> {
        // 1. Classifica workload com base na telemetria
        let class = classify_workload(&telemetry);

        // 2. Admissão
        if !self.can_admit(model_size_bytes, &class) {
            return None;
        }

        let id = self.next_session_id;
        self.next_session_id += 1;

        self.sessions.push(SessionRecord {
            id,
            class,
            model_name: model_name.to_string(),
            generation: 0,
            last_gap_ms: telemetry.gap_from_last_ms,
            total_tokens: telemetry.total_processed,
            created_at: Instant::now(),
            last_active: Instant::now(),
        });

        // 3. Garante que o modelo está quente (ou agenda swap via swapper)
        if let Some(entry) = self.registry.get(model_name) {
            self.swapper.touch(model_name);
            if entry.tier == MemoryTier::Vram {
                // já quente — registra aquisição
                self.registry.acquire(model_name);
            }
            // Se não está na VRAM, o SwampVM vai fazer swap no primeiro opcode
        }

        Some(id)
    }

    pub fn unregister_session(&mut self, id: SessionId) {
        if let Some(pos) = self.sessions.iter().position(|s| s.id == id) {
            let session = self.sessions.remove(pos);
            self.registry.release(&session.model_name);
        }
    }

    pub fn touch_session(&mut self, id: SessionId) {
        if let Some(s) = self.sessions.iter_mut().find(|s| s.id == id) {
            s.last_active = Instant::now();
            s.generation += 1;
        }
    }

    /// Retorna a generation atual da sessão (pra detecção de stale opcodes na ring)
    pub fn session_generation(&self, id: SessionId) -> Option<u64> {
        self.sessions.iter()
            .find(|s| s.id == id)
            .map(|s| s.generation)
    }

    // =====================================================================
    // Admissão
    // =====================================================================

    pub fn can_admit(&self, model_size_bytes: usize, class: &WorkloadClass) -> bool {
        if *class == WorkloadClass::Cognitive {
            // Cognitivos quase sempre cabem — CPU-only, footprint pequeno
            return model_size_bytes < 500_000_000; // <500MB
        }

        // Verifica VRAM disponível
        let budget = self.aimd.budget();
        if budget < self.min_safe_headroom {
            return false; // sem headroom, não admite mais ninguém
        }

        let active_vram = self.registry.total_vram_active();
        let available = self.swapper.vram_capacity.saturating_sub(active_vram);
        let effective_budget = (budget - self.min_safe_headroom) / budget; // fração usável
        let usable_vram = (available as f64 * effective_budget) as usize;

        // Modelos Batch podem usar uma fração menor (tem fila, pode esperar swap)
        let threshold = match class {
            WorkloadClass::Interactive { .. } => usable_vram,
            WorkloadClass::Batch { .. } => usable_vram / 2, // batch aceita esperar swap
            _ => usable_vram,
        };

        model_size_bytes <= threshold
    }

    // =====================================================================
    // Main cycle — chamado a cada ~500ms (mesmo intervalo do AIMD)
    // =====================================================================

    pub fn cycle(&mut self, pl1_watts: f64) {
        let now = Instant::now();
        let dt = self.last_cycle.elapsed();
        if dt < self.interval {
            return;
        }
        self.last_cycle = now;
        self.cycle_count += 1;

        // 1. Telemetria térmica
        self.thermal.update();
        let temp = self.thermal.current_temp;
        let stressed = self.thermal.predictive_throttle((pl1_watts * 1_000_000.0) as u64)
            || self.thermal.state == crate::thermal::ThermalState::Warning;

        // 2. AIMD ramp (dobra/corta budget)
        self.total_budget = self.aimd.assess(stressed);

        // 3. PowerArbiter split (CPU vs iGPU)
        let (cpu_frac, _igpu_frac) = self.power.reassess(pl1_watts);

        // 4. Auto-tuning das flags
        self.tune_flags(temp, cpu_frac);

        // 5. Evict sessões paradas (>30s sem atividade)
        self.evict_stale_sessions();

        // 6. Prefetch hint pro swapper baseado na fila de sessões
        self.update_swap_hints();
    }

    fn tune_flags(&mut self, temp: f32, cpu_frac: f64) {
        let budget = self.aimd.budget();

        // n_threads: escala com budget * cpu_frac
        let base = self.aimd.scaled_threads(self.max_n_threads);
        self.cpu_threads = (base as f64 * cpu_frac).round().max(1.0) as usize;

        // prefill_chunk_size: AIMD ramp direto
        // budget alto → chunk maior (amortiza overhead)
        // budget baixo → chunk menor (não monopoliza térmico)
        let chunk = (4096.0 * budget).round() as usize;
        self.prefill_chunk_size = chunk.clamp(512, 16384);

        // gpu_model_threshold: ajusta com budget
        // Budget baixo → manda menos pra GPU (economiza energia)
        if budget < 0.5 {
            self.gpu_model_threshold = 4_000_000_000; // essencialmente desliga GPU pra modelos
        } else if budget < 1.0 {
            self.gpu_model_threshold = 3_000_000_000;
        } else {
            self.gpu_model_threshold = 1_500_000_000; // modelos >= 1.5GB vão pra GPU
        }

        // Thermal override: se critical, tudo mínimo
        if temp >= 85.0 {
            self.cpu_threads = 1;
            self.prefill_chunk_size = 512;
            self.gpu_model_threshold = usize::MAX; // nada vai pra GPU
        }
    }

    fn evict_stale_sessions(&mut self) {
        let cutoff = Duration::from_secs(30);
        let now = Instant::now();
        self.sessions.retain(|s| {
            let stale = now.saturating_duration_since(s.last_active) > cutoff;
            if stale {
                self.registry.release(&s.model_name);
            }
            !stale
        });
    }

    fn update_swap_hints(&mut self) {
        // O schedule de swap é feito pelo SwampVM quando encontra modelo não-quente.
        // O Governor só dá hint pro swapper sobre qual sessão deve vir a seguir.
        if let Some(next) = self.sessions.first() {
            self.swapper.predict_next(Some(&next.model_name));
        } else {
            self.swapper.predict_next(None);
        }
    }

    // =====================================================================
    // Backpressure — quem cede quando passa do budget
    // =====================================================================

    /// Retorna ordered list de sessões que devem ceder recursos (mais sacrificáveis primeiro)
    pub fn backpressure_order(&self) -> Vec<SessionId> {
        let mut candidates: Vec<_> = self.sessions.iter().collect();

        // Batch cede primeiro (usuário não espera), depois Interactive, Cognitive nunca cede
        candidates.sort_by_key(|s| match s.class {
            WorkloadClass::Batch { .. } => 0,
            WorkloadClass::Interactive { .. } => 1,
            WorkloadClass::Cognitive => 2, // nunca cedem
        });

        candidates.into_iter()
            .filter(|s| s.class != WorkloadClass::Cognitive)
            .map(|s| s.id)
            .collect()
    }

    // =====================================================================
    // Queries
    // =====================================================================

    pub fn current_load(&self) -> f64 {
        let active = self.sessions.iter()
            .filter(|s| s.class != WorkloadClass::Cognitive)
            .count() as f64;
        let capacity = self.aimd.concurrency() as f64;
        if capacity == 0.0 { return 0.0; }
        active / capacity
    }

    pub fn find_session(&self, id: SessionId) -> Option<&SessionRecord> {
        self.sessions.iter().find(|s| s.id == id)
    }

    pub fn is_stale_generation(&self, id: SessionId, generation: u64) -> bool {
        self.sessions.iter()
            .find(|s| s.id == id)
            .map_or(true, |s| s.generation != generation)
    }
}

// =====================================================================
// Workload classification
// =====================================================================

fn classify_workload(tel: &SessionTelemetry) -> WorkloadClass {
    // Gap muito longo ou primeira vez → interativo (conservador)
    if tel.total_processed < 5 {
        return WorkloadClass::Interactive { avg_gap_ms: tel.gap_from_last_ms };
    }

    // Fila: gap < 50ms por request significa fila acumulando
    if tel.gap_from_last_ms < 50.0 {
        return WorkloadClass::Batch { queue_depth: 1 };
    }

    // Gaps curtos (~200-500ms) e prompts pequenos → interativo
    if tel.gap_from_last_ms < 1000.0 && tel.prompt_len < 512 {
        return WorkloadClass::Interactive { avg_gap_ms: tel.gap_from_last_ms };
    }

    // Gaps longos e prompts grandes → batch
    WorkloadClass::Batch { queue_depth: 1 }
}

// =====================================================================
// CognitiveWorkerPool — modelos pequenos CPU-only sem SwampVM overhead
// =====================================================================

/// Pool separado pra workloads cognitivas (classificadores, DSPark, roteador).
/// Roda direto nos threads do RAYON_POOL sem passar pela ring do SwampVM.
pub struct CognitiveWorkerPool {
    max_threads: usize,
    active: Arc<std::sync::atomic::AtomicU32>,
}

impl CognitiveWorkerPool {
    pub fn new(max_threads: usize) -> Self {
        Self {
            max_threads,
            active: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    /// Tenta adquirir um slot de execução. Retorna None se lotado.
    pub fn try_acquire(&self) -> Option<CognitiveToken> {
        loop {
            let current = self.active.load(std::sync::atomic::Ordering::Relaxed);
            if current >= self.max_threads as u32 {
                return None;
            }
            if self.active.compare_exchange_weak(
                current, current + 1,
                std::sync::atomic::Ordering::Acquire,
                std::sync::atomic::Ordering::Relaxed,
            ).is_ok() {
                return Some(CognitiveToken { active: self.active.clone() });
            }
        }
    }
}

pub struct CognitiveToken {
    active: Arc<std::sync::atomic::AtomicU32>,
}

impl Drop for CognitiveToken {
    fn drop(&mut self) {
        self.active.fetch_sub(1, std::sync::atomic::Ordering::Release);
    }
}
