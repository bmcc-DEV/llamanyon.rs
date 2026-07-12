use ash::vk;
use std::sync::Arc;
use crate::vulkan::VkBackend;

#[derive(Clone)]
pub enum ComputeNodeOp {
    GEMVQ4K {
        d_w: vk::Buffer,
        d_x: vk::Buffer,
        d_out: vk::Buffer,
        n_rows: u32,
        n_blocks: u32,
    },
    Attention {
        d_q: vk::Buffer,
        d_k: vk::Buffer,
        d_v: vk::Buffer,
        d_scores: vk::Buffer,
        d_out: vk::Buffer,
        n_heads: u32,
        n_kv_heads: u32,
        seq_len: u32,
        head_dim: u32,
        kv_stride: u32,
    },
    RMSNorm {
        d_x: vk::Buffer,
        d_w: vk::Buffer,
        d_out: vk::Buffer,
        n: u32,
        eps: f32,
    },
    RoPE {
        d_q: vk::Buffer,
        d_k: vk::Buffer,
        pos: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
    },
    SiLUMul {
        d_gate: vk::Buffer,
        d_up: vk::Buffer,
        d_out: vk::Buffer,
        n: u32,
    },
    Add {
        d_a: vk::Buffer,
        d_b: vk::Buffer,
        d_out: vk::Buffer,
        n: u32,
    },
    MemoryBarrier,
}

pub struct ComputeNode {
    pub op: ComputeNodeOp,
    pub dependencies: Vec<usize>,
}

unsafe impl Send for ComputeNode {}
unsafe impl Sync for ComputeNode {}

pub struct ComputeGraph {
    backend: Arc<VkBackend>,
    nodes: Vec<ComputeNode>,
    command_buffer: Option<vk::CommandBuffer>,
    fence: vk::Fence,
    timeline_semaphore: vk::Semaphore,
    pub timeline_value: u64,
}

impl ComputeGraph {
    pub fn new(backend: &Arc<VkBackend>) -> Result<Self, crate::vulkan::VkError> {
        let mut timeline_type = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);

        let semaphore_create_info = vk::SemaphoreCreateInfo::default()
            .push_next(&mut timeline_type);

        let timeline_semaphore = unsafe {
            backend.device.create_semaphore(&semaphore_create_info, None)?
        };

        let fence = unsafe {
            backend.device.create_fence(
                &vk::FenceCreateInfo::default()
                    .flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?
        };

        Ok(Self {
            backend: backend.clone(),
            nodes: Vec::with_capacity(64),
            command_buffer: None,
            fence,
            timeline_semaphore,
            timeline_value: 0,
        })
    }

    pub fn add_node(&mut self, op: ComputeNodeOp, dependencies: Vec<usize>) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(ComputeNode { op, dependencies });
        idx
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn build(&mut self, pool: vk::CommandPool) -> Result<vk::CommandBuffer, crate::vulkan::VkError> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffers = unsafe {
            self.backend.device.allocate_command_buffers(&allocate_info)?
        };
        let command_buffer = cmd_buffers[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

        unsafe {
            self.backend.device.begin_command_buffer(command_buffer, &begin_info)?;
        }

        for node in &self.nodes {
            if !node.dependencies.is_empty() {
                let barriers: Vec<vk::BufferMemoryBarrier> = node.dependencies.iter()
                    .filter_map(|dep_idx| {
                        let dep = self.nodes.get(*dep_idx)?;
                        let buf = match &dep.op {
                            ComputeNodeOp::GEMVQ4K { d_out, .. } => *d_out,
                            ComputeNodeOp::Attention { d_out, .. } => *d_out,
                            ComputeNodeOp::RMSNorm { d_out, .. } => *d_out,
                            _ => return None,
                        };
                        (buf != vk::Buffer::null()).then(|| {
                            vk::BufferMemoryBarrier::default()
                                .buffer(buf)
                                .size(vk::WHOLE_SIZE)
                                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        })
                    })
                    .collect();

                if !barriers.is_empty() {
                    unsafe {
                        self.backend.device.cmd_pipeline_barrier(
                            command_buffer,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &barriers,
                            &[],
                        );
                    }
                }
            }

            if let ComputeNodeOp::MemoryBarrier = &node.op {
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                unsafe {
                    self.backend.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    );
                }
            }
        }

        unsafe {
            self.backend.device.end_command_buffer(command_buffer)?;
        }

        self.command_buffer = Some(command_buffer);
        Ok(command_buffer)
    }

    pub fn submit(&self) -> Result<(), crate::vulkan::VkError> {
        let cb = self.command_buffer
            .ok_or_else(|| crate::vulkan::VkError::NotAvailable("Graph not built".into()))?;

        unsafe { self.backend.device.reset_fences(&[self.fence])?; }

        let cbs = [cb];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cbs);
        let submits = [submit_info];

        unsafe {
            self.backend.device.queue_submit(
                self.backend._queue,
                &submits,
                self.fence,
            )?;
        }

        Ok(())
    }

    pub fn wait(&self) -> Result<(), crate::vulkan::VkError> {
        unsafe {
            self.backend.device.wait_for_fences(
                &[self.fence],
                true,
                10_000_000_000,
            )?;
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
        self.command_buffer = None;
    }
}

unsafe impl Send for ComputeGraph {}
unsafe impl Sync for ComputeGraph {}

impl Drop for ComputeGraph {
    fn drop(&mut self) {
        unsafe {
            if self.fence != vk::Fence::null() {
                self.backend.device.destroy_fence(self.fence, None);
            }
            if self.timeline_semaphore != vk::Semaphore::null() {
                self.backend.device.destroy_semaphore(self.timeline_semaphore, None);
            }
        }
    }
}
