use libloading::{Library, Symbol};
use std::sync::OnceLock;
use tracing::info;

type SparseAttentionFn = unsafe extern "C" fn(u64, u64, u64, u64, u64, i32, i32, i32, i32, i32);

pub struct MojoKernel {
    lib: Option<Library>,
    sparse_attention: Option<SparseAttentionFn>,
}

impl MojoKernel {
    pub fn new() -> Self {
        match Self::try_load() {
            Ok(k) => {
                info!("Mojo kernel library loaded (libswamp_mojo.so)");
                k
            }
            Err(e) => {
                tracing::warn!("Mojo kernel not available: {}", e);
                Self::dummy()
            }
        }
    }

    fn try_load() -> Result<Self, Box<dyn std::error::Error>> {
        let paths = [
            "libswamp_mojo.so",
            "./swamp-gpu/libswamp_mojo.so",
            "../swamp-gpu/libswamp_mojo.so",
        ];

        for path in &paths {
            match unsafe { Library::new(path) } {
                Ok(lib) => {
                    let sparse_attention: Symbol<SparseAttentionFn> = unsafe {
                        lib.get(b"sparse_attention_q4")?
                    };
                    let func_ptr = *sparse_attention;
                    return Ok(Self {
                        lib: Some(lib),
                        sparse_attention: Some(func_ptr),
                    });
                }
                Err(e) => {
                    tracing::debug!("Mojo lib not found at {}: {}", path, e);
                }
            }
        }
        Err("no mojo library found".into())
    }

    fn dummy() -> Self {
        Self {
            lib: None,
            sparse_attention: None,
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.sparse_attention.is_some()
    }

    pub fn sparse_attention(
        &self,
        q: &[f32],
        k_q4: &[f32],
        v_q4: &[f32],
        positions: &[i32],
        out: &mut [f32],
        n_heads: i32,
        head_dim: i32,
        n_sparse: i32,
        window: i32,
    ) -> bool {
        let f = match self.sparse_attention {
            Some(f) => f,
            None => return false,
        };
        unsafe {
            f(
                q.as_ptr() as u64,
                k_q4.as_ptr() as u64,
                v_q4.as_ptr() as u64,
                positions.as_ptr() as u64,
                out.as_mut_ptr() as u64,
                n_heads,
                head_dim,
                n_sparse,
                window,
                n_heads * head_dim * 4,
            );
        }
        true
    }
}

static MOJO: OnceLock<MojoKernel> = OnceLock::new();

pub fn get_mojo() -> &'static MojoKernel {
    MOJO.get_or_init(MojoKernel::new)
}
