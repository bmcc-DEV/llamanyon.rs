// swamp-engine/src/staging.rs
// StagingBuffer — workers desacoplados (CPU threads, iGPU, dGPU) produzem
// resultados parciais em RAM compartilhada, sem sincronização apertada.
//
// Cada worker recebe um slice do trabalho (ex: N heads da attention,
// M linhas do GEMV) e escreve seu resultado no slice correspondente.
// O coordenador espera todos os slices ficarem prontos e combina.

use std::sync::atomic::{AtomicBool, Ordering};

pub struct StagingSlice {
    pub data: Vec<f32>,
    ready: AtomicBool,
}

impl StagingSlice {
    fn new(size: usize) -> Self {
        Self {
            data: vec![0.0f32; size],
            ready: AtomicBool::new(false),
        }
    }

    pub fn mark_ready(&self) {
        self.ready.store(true, Ordering::Release);
    }

    pub fn wait_ready(&self) {
        while !self.ready.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
    }

    pub fn reset(&self) {
        self.ready.store(false, Ordering::Release);
    }
}

pub struct StagingBuffer {
    slices: Vec<StagingSlice>,
    slice_size: usize,
    num_slices: usize,
}

impl StagingBuffer {
    /// Create a staging buffer with `num_slices` slices, each of `slice_size` elements.
    pub fn new(num_slices: usize, slice_size: usize) -> Self {
        let slices = (0..num_slices)
            .map(|_| StagingSlice::new(slice_size))
            .collect();
        Self {
            slices,
            slice_size,
            num_slices,
        }
    }

    pub fn num_slices(&self) -> usize {
        self.num_slices
    }

    pub fn slice_size(&self) -> usize {
        self.slice_size
    }

    /// Get a read-only view into slice `i`.
    pub fn slice_data(&self, i: usize) -> &[f32] {
        &self.slices[i].data
    }

    /// Get a mutable view into slice `i` (for the coordinating thread).
    pub fn slice_data_mut(&mut self, i: usize) -> &mut [f32] {
        &mut self.slices[i].data
    }

    /// Reset all slices (set not-ready + zero).
    pub fn reset_all(&mut self) {
        for slice in &mut self.slices {
            slice.data.fill(0.0);
            slice.reset();
        }
    }

    /// Wait for all slices to be ready.
    pub fn wait_all(&self) {
        for slice in &self.slices {
            slice.wait_ready();
        }
    }

    /// Combine all slices into output via element-wise sum.
    pub fn combine_into(&self, output: &mut [f32]) {
        output.copy_from_slice(&self.slices[0].data);
        for slice in &self.slices[1..] {
            for (o, s) in output.iter_mut().zip(slice.data.iter()) {
                *o += s;
            }
        }
    }
}
