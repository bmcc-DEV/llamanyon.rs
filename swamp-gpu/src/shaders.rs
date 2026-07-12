use std::sync::Arc;
use ash::vk;
use crate::vulkan::VkBackend;

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum ShaderType {
    GEMVQ4K,
    GEMVQ6K,
    Attention,
    RMSNorm,
    RoPE,
    SiLUMul,
    Add,
}

impl ShaderType {
    pub fn name(&self) -> &'static str {
        match self {
            ShaderType::GEMVQ4K => "gemv_q4k",
            ShaderType::GEMVQ6K => "gemv_q6k",
            ShaderType::Attention => "attention",
            ShaderType::RMSNorm => "rmsnorm",
            ShaderType::RoPE => "rope",
            ShaderType::SiLUMul => "silu_mul",
            ShaderType::Add => "add",
        }
    }
}

pub struct ShaderCache {
    backend: Arc<VkBackend>,
    modules: Vec<(ShaderType, vk::ShaderModule)>,
    pipelines: Vec<(ShaderType, vk::Pipeline, vk::PipelineLayout)>,
}

impl ShaderCache {
    pub fn new(backend: &Arc<VkBackend>) -> Self {
        Self {
            backend: backend.clone(),
            modules: Vec::new(),
            pipelines: Vec::new(),
        }
    }

    pub fn load_shader(&mut self, shader: ShaderType, spirv: &[u32]) -> Result<vk::ShaderModule, crate::vulkan::VkError> {
        if let Some(&module) = self.modules.iter()
            .find(|(t, _)| *t == shader)
            .map(|(_, m)| m)
        {
            return Ok(module);
        }
        let module = self.backend.create_shader_module(spirv)?;
        self.modules.push((shader, module));
        Ok(module)
    }

    pub fn get_or_create_pipeline(
        &mut self,
        shader: ShaderType,
        spirv: &[u32],
        push_constant_size: u32,
    ) -> Result<vk::Pipeline, crate::vulkan::VkError> {
        for &(ref t, p, _) in &self.pipelines {
            if *t == shader {
                return Ok(p);
            }
        }

        let module = self.load_shader(shader, spirv)?;

        let push_constants = if push_constant_size > 0 {
            vec![vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: push_constant_size,
            }]
        } else {
            vec![]
        };

        let layout = unsafe {
            self.backend.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .push_constant_ranges(&push_constants),
                None,
            )?
        };

        let pipeline = self.backend.create_compute_pipeline(
            module,
            &push_constants,
            &[],
        )?;
        self.pipelines.push((shader, pipeline, layout));
        Ok(pipeline)
    }

    pub fn pipeline(&self, shader: &ShaderType) -> Option<vk::Pipeline> {
        self.pipelines.iter()
            .find(|(t, _, _)| *t == *shader)
            .map(|(_, p, _)| *p)
    }
}

impl Drop for ShaderCache {
    fn drop(&mut self) {
        if !self.backend.enabled { return; }
        unsafe {
            for (_, pipeline, layout) in self.pipelines.drain(..) {
                self.backend.device.destroy_pipeline(pipeline, None);
                self.backend.device.destroy_pipeline_layout(layout, None);
            }
            for (_, module) in self.modules.drain(..) {
                self.backend.device.destroy_shader_module(module, None);
            }
        }
    }
}
