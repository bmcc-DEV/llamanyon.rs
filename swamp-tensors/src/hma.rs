use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    HostPinned,
    DeviceLocal,
    UnifiedMapped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Q4K,
    Q6K,
}

impl DType {
    pub fn element_size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::Q4K => 1,
            DType::Q6K => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Allocation {
    pub ptr: *mut u8,
    pub size: usize,
    pub location: MemoryLocation,
    pub device_handle: Option<u64>,
}

unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Allocation {
    pub fn null() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            size: 0,
            location: MemoryLocation::HostPinned,
            device_handle: None,
        }
    }

    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }

    pub fn as_slice_f32(&self) -> &[f32] {
        assert!(!self.ptr.is_null());
        assert_eq!(self.size % 4, 0);
        unsafe { std::slice::from_raw_parts(self.ptr as *const f32, self.size / 4) }
    }

    pub fn as_slice_f32_mut(&mut self) -> &mut [f32] {
        assert!(!self.ptr.is_null());
        assert_eq!(self.size % 4, 0);
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut f32, self.size / 4) }
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.location == MemoryLocation::HostPinned {
            unsafe {
                let _ = Vec::from_raw_parts(self.ptr, 0, self.size);
            }
        }
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct TensorHandle {
    pub id: u64,
    pub location: MemoryLocation,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub allocation: Option<Allocation>,
}

impl TensorHandle {
    pub fn new(location: MemoryLocation, dtype: DType, shape: Vec<usize>) -> Self {
        let strides = compute_strides(&shape);
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            location,
            dtype,
            shape,
            strides,
            allocation: None,
        }
    }

    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.element_count() * self.dtype.element_size()
    }

    pub fn allocate(&mut self, allocator: &HeterogeneousMemoryAllocator) -> bool {
        let size = self.byte_size();
        let alloc = allocator.allocate(size, self.location);
        match alloc {
            Some(a) => {
                self.allocation = Some(a);
                true
            }
            None => false,
        }
    }

    pub fn device_id(&self) -> Option<u64> {
        self.allocation.as_ref().and_then(|a| a.device_handle)
    }

    pub fn host_ptr(&self) -> *mut u8 {
        self.allocation.as_ref().map_or(std::ptr::null_mut(), |a| a.ptr)
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        strides.push(stride);
        stride *= shape[i];
    }
    strides.reverse();
    strides
}

struct HostPinnedArenaInner {
    blocks: Vec<Allocation>,
    bump_offset: usize,
}

pub struct HostPinnedArena {
    inner: Mutex<HostPinnedArenaInner>,
    block_size: usize,
}

impl HostPinnedArena {
    pub fn new(block_size: usize) -> Self {
        let block_size = block_size.next_power_of_two().max(4096);
        Self {
            inner: Mutex::new(HostPinnedArenaInner {
                blocks: Vec::new(),
                bump_offset: 0,
            }),
            block_size,
        }
    }

    pub fn allocate(&self, size: usize) -> Option<Allocation> {
        let size = size.next_power_of_two().max(64);
        let mut inner = self.inner.lock();
        let need_new = {
            let last_size = inner.blocks.last().map_or(0, |b| b.size);
            inner.bump_offset + size > last_size
        };
        if need_new {
            let alloc_size = size.max(self.block_size);
            let mut vec: Vec<u8> = Vec::with_capacity(alloc_size);
            let ptr = vec.as_mut_ptr();
            std::mem::forget(vec);
            inner.blocks.push(Allocation {
                ptr,
                size: alloc_size,
                location: MemoryLocation::HostPinned,
                device_handle: None,
            });
            inner.bump_offset = 0;
        }
        let offset = inner.bump_offset;
        inner.bump_offset += size;
        let block_ptr = inner.blocks.last()?.ptr;
        Some(Allocation {
            ptr: unsafe { block_ptr.add(offset) },
            size,
            location: MemoryLocation::HostPinned,
            device_handle: None,
        })
    }

    pub fn reset(&self) {
        let mut inner = self.inner.lock();
        inner.bump_offset = 0;
    }
}

impl Drop for HostPinnedArena {
    fn drop(&mut self) {
        let inner = self.inner.lock();
        for block in &inner.blocks {
            if !block.ptr.is_null() {
                unsafe {
                    let _ = Vec::from_raw_parts(block.ptr, 0, block.size);
                }
            }
        }
    }
}

pub trait DeviceAllocator: Send + Sync {
    fn allocate_device(&self, size: usize) -> Option<Allocation>;
    fn allocate_unified(&self, size: usize) -> Option<Allocation>;
    fn free(&self, alloc: &Allocation);
    fn upload(&self, dst: &Allocation, src: &[u8]);
    fn download(&self, dst: &mut [u8], src: &Allocation);
}

pub struct HeterogeneousMemoryAllocator {
    pub host: HostPinnedArena,
    device: Option<Box<dyn DeviceAllocator>>,
}

impl HeterogeneousMemoryAllocator {
    pub fn new() -> Self {
        Self {
            host: HostPinnedArena::new(16 * 1024 * 1024),
            device: None,
        }
    }

    pub fn with_device(device: Box<dyn DeviceAllocator>) -> Self {
        Self {
            host: HostPinnedArena::new(16 * 1024 * 1024),
            device: Some(device),
        }
    }

    pub fn allocate(&self, size: usize, location: MemoryLocation) -> Option<Allocation> {
        match location {
            MemoryLocation::HostPinned => self.host.allocate(size),
            MemoryLocation::DeviceLocal | MemoryLocation::UnifiedMapped => {
                match &self.device {
                    Some(dev) => match location {
                        MemoryLocation::DeviceLocal => dev.allocate_device(size),
                        MemoryLocation::UnifiedMapped => dev.allocate_unified(size),
                        _ => unreachable!(),
                    },
                    None => {
                        tracing::warn!("Device allocator not available, falling back to host");
                        self.host.allocate(size)
                    }
                }
            }
        }
    }

    pub fn reset(&self) {
        self.host.reset();
    }
}

impl Default for HeterogeneousMemoryAllocator {
    fn default() -> Self {
        Self::new()
    }
}
