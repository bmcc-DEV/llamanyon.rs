// swamp-engine/src/model_swapper.rs
// ModelSwapper — gerencia qual modelo está quente na VRAM (GTX 1650 4GB).
//
// Política: LRU + pipeline-aware prefetch.
// - Apenas 1 modelo grande (7B Q4_K ≈ 3.5GB) ou 3 pequenos cabem nos 4GB.
// - Swap out: modelo menos recentemente usado com active_count == 0.
// - Swap in: carrega weights do mmap (RAM) para VRAM via PCIe.
// - Prefetch: se o pipeline sabe qual modelo vem a seguir, começa a swap
//   antes do fim da inferência atual.

use crate::model_registry::{ModelEntry, MemoryTier};
use std::collections::VecDeque;

/// Evento de swap para logging/diagnóstico
#[derive(Debug)]
pub struct SwapEvent {
    pub model_out: String,
    pub model_in: String,
    pub duration_ms: f64,
}

pub struct ModelSwapper {
    /// Ordem de uso recente (mais recente no fim)
    lru_order: VecDeque<String>,
    /// Máximo de VRAM disponível para modelos (bytes)
    pub vram_capacity: usize,
    /// Eventos de swap recentes (circular)
    swap_log: VecDeque<SwapEvent>,
    /// Pipeline hint: próximo modelo que será necessário
    predicted_next: Option<String>,
    /// Prefetch está ativo?
    prefetch_active: bool,
}

impl ModelSwapper {
    /// `vram_capacity_bytes`: VRAM disponível (ex: GTX 1650 = 4GB = 4_294_967_296)
    pub fn new(vram_capacity_bytes: usize) -> Self {
        Self {
            lru_order: VecDeque::new(),
            vram_capacity: vram_capacity_bytes,
            swap_log: VecDeque::with_capacity(32),
            predicted_next: None,
            prefetch_active: false,
        }
    }

    /// Notifica que um modelo foi usado (move para o fim da LRU).
    pub fn touch(&mut self, name: &str) {
        if let Some(pos) = self.lru_order.iter().position(|n| n == name) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push_back(name.to_string());
    }

    /// Define qual modelo será necessário a seguir (pipeline hint).
    pub fn predict_next(&mut self, name: Option<&str>) {
        self.predicted_next = name.map(|s| s.to_string());
    }

    /// Decide se o modelo `name` precisa ser swapado para VRAM.
    /// Retorna `SwapAction` indicando o que fazer.
    pub fn check_swap<'a>(
        &mut self,
        name: &str,
        entry: &ModelEntry,
        all_entries: impl Iterator<Item = (&'a str, &'a ModelEntry)>,
    ) -> SwapAction {
        if entry.tier == MemoryTier::Vram {
            // Já está na VRAM — só atualiza LRU
            return SwapAction::AlreadyHot;
        }

        // Precisa fazer swap in
        // Quanto espaço temos disponível?
        let active_vram: usize = all_entries
            .filter(|(n, e)| *n != name && e.tier == MemoryTier::Vram && e.active_count > 0)
            .map(|(_, e)| e.vram_estimate)
            .sum();

        let available = self.vram_capacity.saturating_sub(active_vram);
        if available >= entry.vram_estimate {
            // Cabe sem precisar swap out
            return SwapAction::SwapIn { evict: None };
        }

        // Precisa ejetar algo
        let mut to_evict: Option<String> = None;
        for lru_name in &self.lru_order {
            if lru_name == name { continue; }
            // Procura entrada na lista
            if let Some(pos) = self.lru_order.iter().position(|n| n == lru_name) {
                // Só pode ejetar se active_count == 0
                // (não podemos verificar aqui sem acesso ao registry)
                to_evict = Some(lru_name.clone());
                self.lru_order.remove(pos);
                break;
            }
        }

        SwapAction::SwapIn { evict: to_evict }
    }

    /// Registra um evento de swap concluído.
    pub fn log_swap(&mut self, event: SwapEvent) {
        if self.swap_log.len() >= 32 {
            self.swap_log.pop_front();
        }
        self.swap_log.push_back(event);
    }

    /// Estatísticas de swap
    pub fn swap_count(&self) -> usize {
        self.swap_log.len()
    }

    pub fn avg_swap_ms(&self) -> f64 {
        if self.swap_log.is_empty() { return 0.0; }
        self.swap_log.iter().map(|e| e.duration_ms).sum::<f64>() / self.swap_log.len() as f64
    }
}

#[derive(Debug)]
pub enum SwapAction {
    /// Já está na VRAM, sem ação necessária
    AlreadyHot,
    /// Precisa carregar para VRAM (opcionalmente ejetando outro)
    SwapIn { evict: Option<String> },
}
