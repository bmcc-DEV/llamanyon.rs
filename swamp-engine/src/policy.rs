use mlua::{Lua, Result as LuaResult, Value, UserData, UserDataMethods};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::path::Path;
use std::time::UNIX_EPOCH;

use crate::control_plane::{CommandQueue, Command, ResourceHandle, TensorCreateCmd};
use swamp_tensors::hma::DType;

pub struct PolicyEngine {
    lua: Arc<std::sync::Mutex<Lua>>,
    script_path: String,
    last_load_ns: AtomicI64,
    fallback_threads: usize,
    command_queue: Option<Arc<CommandQueue>>,
}

impl PolicyEngine {
    pub fn new(script_path: &str, fallback_threads: usize) -> LuaResult<Self> {
        let lua = Lua::new();

        lua.load("os = nil; io = nil; debug = nil").exec()?;

        let engine = Self {
            lua: Arc::new(std::sync::Mutex::new(lua)),
            script_path: script_path.to_string(),
            last_load_ns: AtomicI64::new(0),
            fallback_threads,
            command_queue: None,
        };

        if !script_path.is_empty() {
            engine.reload()?;
        }
        Ok(engine)
    }

    pub fn set_command_queue(&mut self, cq: Arc<CommandQueue>) -> LuaResult<()> {
        self.command_queue = Some(cq.clone());
        let lua = self.lua.lock().unwrap();

        let gpu_module = lua.create_table()?;
        let tensor_table = lua.create_table()?;

        let cq_clone = cq.clone();
        let create_fn = lua.create_function(move |_, (shape, dtype_str): (Vec<f64>, String)| {
            let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
            let dtype = match dtype_str.as_str() {
                "f32" => DType::F32,
                "f16" => DType::F16,
                "q4k" => DType::Q4K,
                "q6k" => DType::Q6K,
                _ => DType::F32,
            };
            let id = ResourceHandle::new();
            let mut dims = [0usize; 4];
            for (i, &d) in shape_usize.iter().enumerate().take(4) {
                dims[i] = d;
            }
            cq_clone.push(Command::TensorCreate(TensorCreateCmd {
                tag: 1,
                id: id.0,
                dims,
                ndim: shape_usize.len() as u32,
                dtype: dtype as u32,
                location: 0,
                _pad: 0,
            }));
            Ok(id.0 as f64)
        })?;
        tensor_table.set("create", create_fn)?;
        gpu_module.set("tensor", tensor_table)?;

        let cq_clone2 = cq.clone();
        let commit_fn = lua.create_function(move |_, ()| {
            static FRAME_ID: AtomicI64 = AtomicI64::new(0);
            let frame_id = FRAME_ID.fetch_add(1, Ordering::Relaxed);
            cq_clone2.push(Command::Commit(super::control_plane::CommitCmd {
                tag: 6,
                frame_id: frame_id as u64,
                timestamp: std::time::SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0),
                _pad: 0,
            }));
            Ok(())
        })?;
        gpu_module.set("commit", commit_fn)?;

        lua.globals().set("gpu", gpu_module)?;
        Ok(())
    }

    pub fn reload(&self) -> LuaResult<()> {
        let script = std::fs::read_to_string(&self.script_path)
            .map_err(|e| mlua::Error::RuntimeError(format!("Failed to read policy: {}", e)))?;

        let lua = self.lua.lock().unwrap();
        lua.load(&script).exec()?;
        self.update_last_load();
        Ok(())
    }

    pub fn try_reload(&self) {
        if self.script_path.is_empty() {
            return;
        }
        let path = Path::new(&self.script_path);
        if let Ok(metadata) = path.metadata() {
            if let Ok(modified) = metadata.modified() {
                let modified_ns = modified.duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos() as i64)
                    .unwrap_or(0);
                if modified_ns > self.last_load_ns.load(Ordering::Acquire) {
                    if let Err(e) = self.reload() {
                        eprintln!("[swamp] Policy reload failed: {}", e);
                    } else {
                        println!("[swamp] Policy hot-reloaded: {}", self.script_path);
                    }
                }
            }
        }
    }

    fn update_last_load(&self) {
        if self.script_path.is_empty() {
            return;
        }
        if let Ok(metadata) = Path::new(&self.script_path).metadata() {
            if let Ok(modified) = metadata.modified() {
                if let Ok(d) = modified.duration_since(UNIX_EPOCH) {
                    self.last_load_ns.store(d.as_nanos() as i64, Ordering::Release);
                }
            }
        }
    }

    pub fn adapt_threads(&self, c_epsilon: f64, temp_celsius: f64, freq_mhz: u64) -> usize {
        let lua = self.lua.lock().unwrap();

        match lua.globals().get::<Value>("adapt_threads") {
            Ok(Value::Function(f)) => {
                match f.call::<i64>((c_epsilon, temp_celsius, freq_mhz as i64)) {
                    Ok(n) => (n.max(1).min(6)) as usize,
                    Err(e) => {
                        eprintln!("[swamp] Lua adapt_threads error: {}", e);
                        self.fallback_threads
                    }
                }
            }
            _ => self.fallback_threads,
        }
    }

    pub fn should_skip_layer(&self, layer_idx: usize, temp_celsius: f64) -> bool {
        let lua = self.lua.lock().unwrap();

        match lua.globals().get::<Value>("should_skip_layer") {
            Ok(Value::Function(f)) => {
                match f.call::<bool>((layer_idx as i64, temp_celsius)) {
                    Ok(skip) => skip,
                    Err(e) => {
                        eprintln!("[swamp] Lua should_skip_layer error: {}", e);
                        false
                    }
                }
            }
            _ => false,
        }
    }

    pub fn script_path(&self) -> &str {
        &self.script_path
    }

    pub fn suggested_batch_size(&self, context_tokens: usize, temp_celsius: f64) -> usize {
        let lua = self.lua.lock().unwrap();

        match lua.globals().get::<Value>("suggest_batch_size") {
            Ok(Value::Function(f)) => {
                match f.call::<i64>((context_tokens as i64, temp_celsius)) {
                    Ok(n) => (n.max(8).min(2048)) as usize,
                    Err(e) => {
                        eprintln!("[swamp] Lua suggest_batch_size error: {}", e);
                        512
                    }
                }
            }
            _ => 512,
        }
    }
}
