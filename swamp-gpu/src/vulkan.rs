use ash::{vk, Instance, Device, Entry};
use std::ffi::CStr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VkError {
    #[error("Vulkan not available: {0}")]
    NotAvailable(String),
    #[error("Vulkan error: {0}")]
    VkError(#[from] vk::Result),
    #[error("Shader compilation failed: {0}")]
    ShaderError(String),
    #[error("Loader error: {0}")]
    LoaderError(String),
}

pub type Result<T> = std::result::Result<T, VkError>;

unsafe impl Send for VkBackend {}
unsafe impl Sync for VkBackend {}

pub struct VkBackend {
    pub _entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub _physical_device: vk::PhysicalDevice,
    pub _queue: vk::Queue,
    pub _queue_family: u32,
    pub _memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub pipeline_cache: vk::PipelineCache,
    pub compute_pool: vk::CommandPool,
    pub _device_properties: vk::PhysicalDeviceProperties,
    pub enabled: bool,
}

impl VkBackend {
    pub fn new() -> Self {
        match Self::init_vulkan() {
            Ok(backend) => {
                let name = backend._device_properties.device_name;
                let name_str = unsafe { CStr::from_ptr(name.as_ptr()) }
                    .to_string_lossy();
                tracing::info!("Vulkan backend initialized: {}", name_str);
                backend
            }
            Err(e) => {
                tracing::warn!("Vulkan backend unavailable: {}", e);
                Self::disabled()
            }
        }
    }

    fn disabled() -> Self {
        unsafe {
            let entry = Entry::load().expect("Vulkan loader not available");
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default(), None)
                .expect("Failed to create Vulkan instance");
            let pds = instance.enumerate_physical_devices().unwrap_or_default();
            let pd = *pds.first().unwrap_or(&vk::PhysicalDevice::null());
            let device = instance
                .create_device(pd, &vk::DeviceCreateInfo::default(), None)
                .unwrap_or_else(|_| {
                    instance.create_device(
                        vk::PhysicalDevice::null(),
                        &vk::DeviceCreateInfo::default(),
                        None,
                    ).unwrap()
                });
            Self {
                _entry: entry,
                instance,
                device,
                _physical_device: vk::PhysicalDevice::null(),
                _queue: vk::Queue::null(),
                _queue_family: 0,
                _memory_properties: vk::PhysicalDeviceMemoryProperties::default(),
                pipeline_cache: vk::PipelineCache::null(),
                compute_pool: vk::CommandPool::null(),
                _device_properties: vk::PhysicalDeviceProperties::default(),
                enabled: false,
            }
        }
    }

    fn init_vulkan() -> Result<Self> {
        let entry = unsafe {
            Entry::load()
                .map_err(|e| VkError::LoaderError(format!("{:?}", e)))?
        };

        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_2);

        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app_info),
                None,
            )?
        };

        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let (physical_device, queue_family) = physical_devices
            .iter()
            .filter_map(|&pd| {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                let queue_families = unsafe {
                    instance.get_physical_device_queue_family_properties(pd)
                };
                let qf = queue_families.iter().position(|qf| {
                    qf.queue_flags.contains(vk::QueueFlags::COMPUTE)
                });
                qf.map(|qf| (pd, qf as u32))
            })
            .next()
            .ok_or_else(|| VkError::NotAvailable("No compute GPU".into()))?;

        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities);

        let device = unsafe {
            instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&[queue_info]),
                None,
            )?
        };

        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };
        let device_properties = unsafe {
            instance.get_physical_device_properties(physical_device)
        };

        let pipeline_cache = unsafe {
            device.create_pipeline_cache(
                &vk::PipelineCacheCreateInfo::default(), None
            )?
        };

        let compute_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };

        Ok(Self {
            _entry: entry,
            instance,
            device,
            _physical_device: physical_device,
            _queue: queue,
            _queue_family: queue_family,
            _memory_properties: memory_properties,
            pipeline_cache,
            compute_pool,
            _device_properties: device_properties,
            enabled: true,
        })
    }

    pub fn find_memory_type(
        &self, type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        for i in 0..self._memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && self._memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return Some(i);
            }
        }
        None
    }

    pub fn create_shader_module(&self, spirv: &[u32]) -> Result<vk::ShaderModule> {
        let module = unsafe {
            self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(spirv),
                None,
            )?
        };
        Ok(module)
    }

    pub fn create_compute_pipeline(
        &self,
        shader_module: vk::ShaderModule,
        push_constant_ranges: &[vk::PushConstantRange],
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
    ) -> Result<vk::Pipeline> {
        let entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_name);

        let layout = unsafe {
            self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .push_constant_ranges(push_constant_ranges)
                    .set_layouts(descriptor_set_layouts),
                None,
            )?
        };

        let pipeline = unsafe {
            let result = self.device.create_compute_pipelines(
                self.pipeline_cache,
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)],
                None,
            );
            match result {
                Ok(mut pipelines) => pipelines.remove(0),
                Err((_partial, e)) => return Err(VkError::VkError(e)),
            }
        };

        Ok(pipeline)
    }

    pub fn allocate_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?
        };

        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mem_type = self.find_memory_type(
            mem_reqs.memory_type_bits,
            memory_properties,
        ).ok_or_else(|| VkError::NotAvailable("No suitable memory type".into()))?;

        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(mem_reqs.size)
                    .memory_type_index(mem_type),
                None,
            )?
        };

        unsafe { self.device.bind_buffer_memory(buffer, memory, 0)? };
        Ok((buffer, memory))
    }
}

impl Drop for VkBackend {
    fn drop(&mut self) {
        if self.enabled {
            unsafe {
                if self.compute_pool != vk::CommandPool::null() {
                    self.device.destroy_command_pool(self.compute_pool, None);
                }
                if self.pipeline_cache != vk::PipelineCache::null() {
                    self.device.destroy_pipeline_cache(self.pipeline_cache, None);
                }
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }
}
