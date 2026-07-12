// swamp-tensors - lib.rs
// Swamp - Tipos de tensor e operacoes CPU

pub mod layout;
pub mod tensor;
pub mod ops;
pub mod arena;
pub mod storage;
pub mod hma;

pub use layout::Layout;
pub use tensor::CpuTensor;
pub use ops::{rms_norm, softmax, rope, matmul};
pub use arena::TensorArena;
pub use storage::LscPrefetcher;
pub use hma::{HeterogeneousMemoryAllocator, HostPinnedArena, DeviceAllocator};
pub use hma::{TensorHandle, Allocation, MemoryLocation, DType};
