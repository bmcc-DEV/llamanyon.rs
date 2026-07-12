use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, TryRecvError};
use std::sync::{Arc, Mutex};
use swamp_tensors::hma::{DType, MemoryLocation};

#[derive(Debug, Clone, Copy)]
#[repr(u64)]
pub enum CommandTag {
    TensorCreate = 1,
    TensorUpload = 2,
    TensorDownload = 3,
    TensorFree = 4,
    KernelDispatch = 5,
    Commit = 6,
    Shutdown = 7,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct TensorCreateCmd {
    pub tag: u64,
    pub id: u64,
    pub dims: [usize; 4],
    pub ndim: u32,
    pub dtype: u32,
    pub location: u32,
    pub _pad: u32,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct TensorUploadCmd {
    pub tag: u64,
    pub id: u64,
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct TensorDownloadCmd {
    pub tag: u64,
    pub id: u64,
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct TensorFreeCmd {
    pub tag: u64,
    pub id: u64,
    pub _pad: [u64; 2],
}

#[derive(Debug, Clone, Copy)]
#[repr(u64)]
pub enum KernelType {
    GEMVQ4K = 0,
    Attention = 1,
    RMSNorm = 2,
    RoPE = 3,
    SiLUMul = 4,
    Add = 5,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct KernelDispatchCmd {
    pub tag: u64,
    pub kernel_type: u64,
    pub kernel_id: u64,
    pub input_ids: [u64; 4],
    pub output_ids: [u64; 2],
    pub n_inputs: u32,
    pub n_outputs: u32,
    pub _pad: u32,
    pub push0: u32, pub push1: u32, pub push2: u32, pub push3: u32,
    pub push4: u32, pub push5: u32, pub push6: u32, pub push7: u32,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct CommitCmd {
    pub tag: u64,
    pub frame_id: u64,
    pub timestamp: u64,
    pub _pad: u64,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct ShutdownCmd {
    pub tag: u64,
    pub _pad: [u64; 3],
}

#[derive(Debug, Clone)]
pub enum Command {
    TensorCreate(TensorCreateCmd),
    TensorUpload(TensorUploadCmd),
    TensorDownload(TensorDownloadCmd),
    TensorFree(TensorFreeCmd),
    KernelDispatch(KernelDispatchCmd),
    Commit(CommitCmd),
    Shutdown(ShutdownCmd),
}

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceHandle(pub u64);

impl ResourceHandle {
    pub fn new() -> Self {
        ResourceHandle(NEXT_HANDLE.fetch_add(1, Ordering::Relaxed))
    }

    pub fn null() -> Self {
        ResourceHandle(0)
    }

    pub fn is_null(&self) -> bool {
        self.0 == 0
    }
}

pub struct CommandQueue {
    sender: mpsc::Sender<Command>,
    receiver: Mutex<mpsc::Receiver<Command>>,
    pending_frames: AtomicU64,
}

impl CommandQueue {
    pub fn new() -> Arc<Self> {
        let (tx, rx) = mpsc::channel();
        Arc::new(Self {
            sender: tx,
            receiver: Mutex::new(rx),
            pending_frames: AtomicU64::new(0),
        })
    }

    pub fn push(&self, cmd: Command) {
        if matches!(&cmd, Command::Commit(_)) {
            self.pending_frames.fetch_add(1, Ordering::Release);
        }
        let _ = self.sender.send(cmd);
    }

    pub fn pop_all(&self) -> Vec<Command> {
        let rx = self.receiver.lock().unwrap();
        let mut batch = Vec::with_capacity(64);
        loop {
            match rx.try_recv() {
                Ok(cmd) => batch.push(cmd),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    batch.push(Command::Shutdown(ShutdownCmd {
                        tag: 7,
                        _pad: [0; 3],
                    }));
                    break;
                }
            }
        }
        batch
    }

    pub fn pending_frames(&self) -> u64 {
        self.pending_frames.load(Ordering::Acquire)
    }

    pub fn pop_frame_batch(&self) -> Vec<Command> {
        let rx = self.receiver.lock().unwrap();
        let mut batch = Vec::with_capacity(32);
        loop {
            match rx.try_recv() {
                Ok(Command::Commit(c)) => {
                    batch.push(Command::Commit(c));
                    let prev = self.pending_frames.fetch_sub(1, Ordering::Release);
                    if prev == 1 {
                        break;
                    }
                }
                Ok(cmd) => batch.push(cmd),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        batch
    }

    pub fn sender(&self) -> mpsc::Sender<Command> {
        self.sender.clone()
    }
}

pub struct OpaqueHandleStore {
    handles: Mutex<std::collections::HashMap<ResourceHandle, OpaqueResource>>,
}

#[derive(Debug, Clone)]
pub enum OpaqueResource {
    Tensor {
        id: u64,
        shape: Vec<usize>,
        dtype: DType,
        location: MemoryLocation,
        byte_size: usize,
        device_handle: u64,
    },
    Kernel {
        name: String,
    },
}

impl OpaqueHandleStore {
    pub fn new() -> Self {
        Self {
            handles: Mutex::new(std::collections::HashMap::new()),
        }
    }

    pub fn register(&self, handle: ResourceHandle, resource: OpaqueResource) {
        let mut map = self.handles.lock().unwrap();
        map.insert(handle, resource);
    }

    pub fn get(&self, handle: ResourceHandle) -> Option<OpaqueResource> {
        let map = self.handles.lock().unwrap();
        map.get(&handle).cloned()
    }

    pub fn remove(&self, handle: ResourceHandle) -> Option<OpaqueResource> {
        let mut map = self.handles.lock().unwrap();
        map.remove(&handle)
    }
}
