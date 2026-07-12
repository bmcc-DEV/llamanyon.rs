use std::process::Command;
use std::path::Path;

fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    // 1. Compile Mojo kernel (attention) → libswamp_mojo.so
    compile_mojo(manifest_dir);

    // 2. Compile GLSL shaders → SPIR-V
    compile_glsl_shaders(manifest_dir);

    // 3. Compile Cap'n Proto schema → Rust bindings
    compile_capnp(manifest_dir);

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/attention.mojo");
    println!("cargo:rerun-if-changed=shaders/");
    println!("cargo:rerun-if-changed=schema/commands.capnp");
}

fn compile_mojo(manifest_dir: &str) {
    let kernel_path = Path::new(manifest_dir).join("kernels/attention.mojo");
    let so_path = Path::new(manifest_dir).join("libswamp_mojo.so");

    let mojo_available = Command::new("which")
        .arg("mojo")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !mojo_available {
        println!("cargo:warning=mojo not found — Mojo attention kernel disabled");
        return;
    }

    println!("cargo:warning=mojo found — building CPU attention kernel");

    let status = Command::new("mojo")
        .args(&[
            "build",
            kernel_path.to_str().unwrap(),
            "-o",
            so_path.to_str().unwrap(),
            "--optimize", "fast",
            "--target", "cpu",
            "--simd-width", "512",
            "--unroll-loops",
        ])
        .status()
        .expect("mojo build failed");

    if !status.success() {
        println!("cargo:warning=mojo build failed — falling back to CPU attention in Rust");
        return;
    }

    println!("cargo:warning=Mojo kernel built: {:?}", so_path);
}

fn compile_glsl_shaders(manifest_dir: &str) {
    let shader_dir = Path::new(manifest_dir).join("shaders");
    let compiled_dir = Path::new(manifest_dir).join("compiled_shaders");

    // Check for GLSL compiler
    let glslc = which_glsl_compiler();
    let glslang = glslc.as_ref().map(|s| s.as_str());

    let compiler = match glslang {
        Some("glslc") => "glslc",
        Some("glslangValidator") => "glslangValidator",
        _ => {
            println!("cargo:warning=No GLSL compiler found (glslc or glslangValidator). GPU shaders disabled.");
            return;
        }
    };

    if !shader_dir.exists() {
        println!("cargo:warning=Shader directory not found: {:?}", shader_dir);
        return;
    }

    std::fs::create_dir_all(&compiled_dir).ok();

    let shaders = [
        "gemv_q4k.comp",
        "attention.comp",
        "rmsnorm.comp",
        "rope.comp",
        "silu_mul.comp",
        "add.comp",
    ];

    for shader in &shaders {
        let src = shader_dir.join(shader);
        let dst_name = format!("{}.spv", shader);
        let dst = compiled_dir.join(&dst_name);

        if !src.exists() {
            println!("cargo:warning=Shader not found: {:?}", src);
            continue;
        }

        let status = if compiler == "glslc" {
            Command::new("glslc")
                .args(&[
                    "-O",
                    src.to_str().unwrap(),
                    "-o",
                    dst.to_str().unwrap(),
                ])
                .status()
        } else {
            Command::new("glslangValidator")
                .args(&[
                    "-V",
                    src.to_str().unwrap(),
                    "-o",
                    dst.to_str().unwrap(),
                ])
                .status()
        };

        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=Shader compiled: {} → {}", shader, dst_name);
            }
            _ => {
                println!("cargo:warning=Shader compilation failed: {}", shader);
            }
        }
    }
}

fn which_glsl_compiler() -> Option<String> {
    for cmd in &["glslc", "glslangValidator"] {
        if Command::new("which")
            .arg(cmd)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return Some(cmd.to_string());
        }
    }
    None
}

fn compile_capnp(manifest_dir: &str) {
    let schema_dir = Path::new(manifest_dir).join("schema");
    let schema_file = schema_dir.join("commands.capnp");

    if !schema_file.exists() {
        println!("cargo:warning=Cap'n Proto schema not found: {:?}", schema_file);
        return;
    }

    let capnp_available = Command::new("which")
        .arg("capnp")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !capnp_available {
        println!("cargo:warning=capnp compiler not found — Cap'n Proto schema disabled");
        return;
    }

    let out_dir = Path::new(manifest_dir).join("src").join("schema_generated");
    std::fs::create_dir_all(&out_dir).ok();

    let status = Command::new("capnp")
        .args(&[
            "compile",
            "-o", "rust",
            &format!("--src-prefix={}", schema_dir.to_str().unwrap()),
            &format!("-o{}", out_dir.to_str().unwrap()),
            schema_file.to_str().unwrap(),
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:warning=Cap'n Proto schema compiled");
        }
        _ => {
            println!("cargo:warning=Cap'n Proto schema compilation failed");
        }
    }
}
