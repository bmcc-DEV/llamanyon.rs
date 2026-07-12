// swamp-engine - lib.rs
// Motor de inferência e controle para LLamañón.rs

pub mod cache;
pub mod executor;
pub mod hlc;
pub mod linear;
pub mod scheduler;
pub mod lsc;
pub mod model;
pub mod prefetch;
pub mod ops;
pub mod tokenizer;
pub mod chat_template;
pub mod sampler;
pub mod thermal;
pub mod vnpu;
pub mod policy;
pub mod dspark;
pub mod fugu;
pub mod timewarp;
pub mod virtual_experts;
pub mod aimd;
pub mod staging;
pub mod power_arbiter;
pub mod model_registry;
pub mod model_swapper;
pub mod governor;
pub mod swamp_vm;
pub mod pipeline;
pub mod control_plane;
pub mod data_plane;

pub use cache::PagedKVCache;
pub use executor::{ModelExecutor, InferenceRequest};
pub use executor::prefill_batch;
pub use linear::{forward_linear, forward_gemvs_ring};
pub use lsc::LscPrefetcher;
pub use model::{Model, ModelConfig};
pub use sampler::Sampler;
pub use thermal::{ThermalCoordinator, ThermalState};
pub use control_plane::{CommandQueue, Command, ResourceHandle, CommandTag};
pub use data_plane::DataPlane;
