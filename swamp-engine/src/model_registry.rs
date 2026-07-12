// swamp-engine/src/model_registry.rs
// ModelRegistry — gerencia múltiplos modelos GGUF carregados em diferentes
// tiers de memória: VRAM (GPU, quente), RAM (pronto pra swap), CPU-only.
//
// Para GTX 1650 4GB: cabe ~1 modelo 7B Q4_K (3.5GB) ou 3 pequenos (~4GB).
// ModelRegistry + ModelSwapper resolvem isso.

use crate::model::Model;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Tier de memória onde o modelo reside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// VRAM da GPU (pronto para inferência GPU acelerada)
    Vram,
    /// RAM do sistema (requer swap para VRAM antes de GPU inference)
    Ram,
    /// CPU-only (nunca vai para GPU, -ngl 0)
    CpuOnly,
}

/// Metadados de um modelo registrado.
#[derive(Clone)]
pub struct ModelEntry {
    pub name: String,
    pub model: Arc<Model>,
    pub tier: MemoryTier,
    /// Tamanho estimado em VRAM (bytes) para agendamento de swap
    pub vram_estimate: usize,
    /// Quantas requisições estão usando este modelo agora
    pub active_count: usize,
}

impl ModelEntry {
    fn vram_estimate(model: &Model) -> usize {
        let gguf = &model.gguf;
        let mmap_len = gguf.mmap_ptr_and_len().1;
        // Q4_K comprime ~4× vs FP32; 1 param FP32 = 4 bytes
        // Estimate: mmap_len já é o tamanho real no disco (já quantizado)
        // Mas VRAM precisa de buffers extras (KV cache, intermediate)
        mmap_len + 512 * 1024 * 1024 // modelo + 512MB folga p/ buffers
    }
}

pub struct ModelRegistry {
    entries: HashMap<String, ModelEntry>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Carrega e registra um modelo GGUF.
    pub fn register<P: AsRef<Path>>(
        &mut self,
        name: &str,
        path: P,
        tier: MemoryTier,
    ) -> anyhow::Result<()> {
        let model = Model::load(path.as_ref())?;
        let vram_estimate = ModelEntry::vram_estimate(&model);
        let entry = ModelEntry {
            name: name.to_string(),
            model: Arc::new(model),
            tier,
            vram_estimate,
            active_count: 0,
        };
        self.entries.insert(name.to_string(), entry);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.entries.get(name)
    }

    pub fn get_model(&self, name: &str) -> Option<Arc<Model>> {
        self.entries.get(name).map(|e| e.model.clone())
    }

    /// Incrementa contagem de uso (para LRU do swapper)
    pub fn acquire(&mut self, name: &str) -> Option<Arc<Model>> {
        let entry = self.entries.get_mut(name)?;
        entry.active_count += 1;
        Some(entry.model.clone())
    }

    pub fn release(&mut self, name: &str) {
        if let Some(entry) = self.entries.get_mut(name) {
            entry.active_count = entry.active_count.saturating_sub(1);
        }
    }

    /// Lista modelos ordenados por prioridade de swap-out (menos ativos primeiro)
    pub fn swap_candidates(&self) -> Vec<(&str, &ModelEntry)> {
        let mut candidates: Vec<_> = self.entries.iter()
            .filter(|(_, e)| e.active_count == 0 && e.tier == MemoryTier::Vram)
            .map(|(k, v)| (k.as_str(), v))
            .collect();
        candidates.sort_by(|a, b| a.1.vram_estimate.cmp(&b.1.vram_estimate));
        candidates
    }

    /// Total estimado de VRAM usada pelos modelos ativos.
    pub fn total_vram_active(&self) -> usize {
        self.entries.values()
            .filter(|e| e.tier == MemoryTier::Vram && e.active_count > 0)
            .map(|e| e.vram_estimate)
            .sum()
    }
}
