use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use parking_lot::Mutex;

use swamp_gpu::{GpuDevice, GpuComputeContext};
use ash::vk;
use vk::Handle;
use swamp_tensors::hma::{HeterogeneousMemoryAllocator, MemoryLocation, DType, TensorHandle};

use crate::control_plane::*;

pub struct DataPlane {
    command_queue: Arc<CommandQueue>,
    handle_store: Arc<OpaqueHandleStore>,
    hma: Arc<Mutex<HeterogeneousMemoryAllocator>>,
    _gpu_device: Arc<GpuDevice>,
    ctx: Option<GpuComputeContext>,
    running: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

unsafe impl Send for DataPlane {}

impl DataPlane {
    pub fn new(
        command_queue: Arc<CommandQueue>,
        handle_store: Arc<OpaqueHandleStore>,
        hma: Arc<Mutex<HeterogeneousMemoryAllocator>>,
        gpu_device: Arc<GpuDevice>,
        embed_dim: usize,
        ffn_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        window_size: usize,
        num_layers: usize,
    ) -> Self {
        let ctx = GpuComputeContext::new(
            &gpu_device, embed_dim, ffn_dim,
            n_heads, n_kv_heads, head_dim, window_size, num_layers,
        );
        Self {
            command_queue,
            handle_store,
            hma,
            _gpu_device: gpu_device,
            ctx,
            running: Arc::new(AtomicBool::new(false)),
            thread: None,
        }
    }

    pub fn start(&mut self) {
        self.running.store(true, Ordering::Release);
        let running = self.running.clone();
        let cq = self.command_queue.clone();
        let hs = self.handle_store.clone();
        let hma = self.hma.clone();

        self.thread = Some(thread::spawn(move || {
            while running.load(Ordering::Acquire) {
                let batch = cq.pop_frame_batch();
                if batch.is_empty() {
                    thread::yield_now();
                    continue;
                }
                Self::process_batch_light(&batch, &hs, &hma);
            }
        }));
    }

    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }

    /// Synchronous batch processing (no thread needed)
    pub fn process_sync(&mut self) {
        let batch = self.command_queue.pop_all();
        if batch.is_empty() { return; }
        if let Some(ref mut ctx) = self.ctx {
            Self::process_batch_with_ctx(&batch, ctx, &self.handle_store, &self.hma);
        }
    }

    /// Light processing: tensor allocation + dispatch without compute (logging)
    fn process_batch_light(
        batch: &[Command],
        hs: &OpaqueHandleStore,
        hma: &Arc<Mutex<HeterogeneousMemoryAllocator>>,
    ) {
        for cmd in batch {
            match cmd {
                Command::TensorCreate(c) => {
                    let ndim = c.ndim as usize;
                    let shape: Vec<usize> = c.dims[..ndim].to_vec();
                    let dtype = match c.dtype {
                        0 => DType::F32, 1 => DType::F16,
                        2 => DType::Q4K, 3 => DType::Q6K,
                        _ => DType::F32,
                    };
                    let location = match c.location {
                        0 => MemoryLocation::HostPinned,
                        1 => MemoryLocation::DeviceLocal,
                        2 => MemoryLocation::UnifiedMapped,
                        _ => MemoryLocation::HostPinned,
                    };
                    let mut tensor = TensorHandle::new(location, dtype, shape);
                    let mut hma_guard = hma.lock();
                    tensor.allocate(&hma_guard);
                    drop(hma_guard);
                    let dev_handle = tensor.allocation.as_ref()
                        .and_then(|a| a.device_handle).unwrap_or(0);
                    hs.register(ResourceHandle(c.id), OpaqueResource::Tensor {
                        id: tensor.id,
                        shape: tensor.shape.clone(),
                        dtype, location,
                        byte_size: tensor.byte_size(),
                        device_handle: dev_handle,
                    });
                    tracing::trace!("[DataPlane] TensorCreate id={} shape={:?}", c.id, tensor.shape);
                }
                Command::KernelDispatch(c) => {
                    tracing::trace!("[DataPlane] KernelDispatch type={} id={}", c.kernel_type, c.kernel_id);
                }
                Command::Commit(c) => {
                    tracing::trace!("[DataPlane] Commit frame={}", c.frame_id);
                }
                Command::Shutdown(_) => {
                    tracing::info!("[DataPlane] Shutdown received");
                    return;
                }
                _ => {}
            }
        }
    }

    /// Full processing with GPU context
    fn process_batch_with_ctx(
        batch: &[Command],
        ctx: &mut GpuComputeContext,
        hs: &OpaqueHandleStore,
        hma: &Arc<Mutex<HeterogeneousMemoryAllocator>>,
    ) {
        for cmd in batch {
            match cmd {
                Command::TensorCreate(c) => {
                    // Same as process_batch_light
                    let ndim = c.ndim as usize;
                    let shape: Vec<usize> = c.dims[..ndim].to_vec();
                    let dtype = match c.dtype {
                        0 => DType::F32, 1 => DType::F16,
                        2 => DType::Q4K, 3 => DType::Q6K,
                        _ => DType::F32,
                    };
                    let location = match c.location {
                        0 => MemoryLocation::HostPinned,
                        1 => MemoryLocation::DeviceLocal,
                        2 => MemoryLocation::UnifiedMapped,
                        _ => MemoryLocation::HostPinned,
                    };
                    let mut tensor = TensorHandle::new(location, dtype, shape);
                    let mut hma_guard = hma.lock();
                    tensor.allocate(&hma_guard);
                    drop(hma_guard);
                    let dev_handle = tensor.allocation.as_ref()
                        .and_then(|a| a.device_handle).unwrap_or(0);
                    hs.register(ResourceHandle(c.id), OpaqueResource::Tensor {
                        id: tensor.id,
                        shape: tensor.shape.clone(),
                        dtype, location,
                        byte_size: tensor.byte_size(),
                        device_handle: dev_handle,
                    });
                }
                Command::KernelDispatch(c) => {
                    if !ctx.is_operational() { continue; }
                    match c.kernel_type {
                        0 => { // GEMVQ4K
                            let w = resolve_buf(c.kernel_id, hs, ctx);
                            let inp = resolve_buf(c.input_ids[0], hs, ctx);
                            let out = resolve_buf(c.output_ids[0], hs, ctx);
                            if let (Some(w), Some(inp), Some(out)) = (w, inp, out) {
                                let n_rows = c.push0;
                                let n_blocks = c.push1;
                                ctx.add_gemv_q4k_node(w, inp, out, n_rows, n_blocks, vec![]);
                            }
                        }
                        1 => { // Attention
                            let d_q = resolve_buf(c.input_ids[0], hs, ctx);
                            let d_k = resolve_buf(c.input_ids[1], hs, ctx);
                            let d_v = resolve_buf(c.input_ids[2], hs, ctx);
                            let d_scores = resolve_buf(c.output_ids[0], hs, ctx);
                            let d_out = resolve_buf(c.output_ids[1], hs, ctx);
                            if let (Some(q), Some(k), Some(v), Some(sc), Some(ot)) = (d_q, d_k, d_v, d_scores, d_out) {
                                ctx.add_attention_node(q, k, v, sc, ot,
                                    c.push0, c.push1, c.push2, c.push3, c.push4, vec![]);
                            }
                        }
                        _ => {
                            tracing::warn!("[DataPlane] Unknown kernel type: {}", c.kernel_type);
                        }
                    }
                }
                Command::Commit(_) => {
                    ctx.submit_and_wait();
                    ctx.reset_graph();
                }
                Command::Shutdown(_) => return,
                _ => {}
            }
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    pub fn command_queue(&self) -> &Arc<CommandQueue> {
        &self.command_queue
    }

    pub fn handle_store(&self) -> &Arc<OpaqueHandleStore> {
        &self.handle_store
    }

    pub fn hma(&self) -> &Arc<Mutex<HeterogeneousMemoryAllocator>> {
        &self.hma
    }
}

/// Resolve a ResourceHandle to a vk::Buffer via handle_store
fn resolve_buf(id: u64, hs: &OpaqueHandleStore, _ctx: &GpuComputeContext) -> Option<vk::Buffer> {
    let handle = ResourceHandle(id);
    let resource = hs.get(handle)?;
    match resource {
        OpaqueResource::Tensor { device_handle, .. } => {
            if device_handle == 0 { return None; }
            Some(unsafe { vk::Buffer::from_raw(device_handle) })
        }
        _ => None,
    }
}

impl Drop for DataPlane {
    fn drop(&mut self) {
        self.stop();
    }
}
