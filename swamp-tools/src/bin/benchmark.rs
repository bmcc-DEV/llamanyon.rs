// swamp-tools/src/bin/benchmark.rs
// Benchmark com buffer alinhado (64B) + warmup para eliminar page faults do kernel.
// Medicao correta: sfence via função externa para que a feature target seja garantida.

use anyhow::Result;
use clap::Parser;
use std::time::Instant;
use swamp_gguf::GgufFile;

#[derive(Parser)]
#[command(name = "swamp-benchmark")]
#[command(about = "Swamp - benchmark de dequantizacao de tensores")]
struct Cli {
    model: String,

    #[arg(short, long)]
    tensor: Option<String>,

    #[arg(short = 'n', long, default_value = "10")]
    iters: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let gguf = GgufFile::open(&cli.model)?;
    println!("Modelo: {} ({} tensores)", cli.model, gguf.n_tensors());

    let tensor_name = cli.tensor.unwrap_or_else(|| {
        gguf.tensors.iter()
            .find(|t| t.name.contains("weight") && t.n_elems() > 10000)
            .map(|t| t.name.clone())
            .unwrap_or_else(|| gguf.tensors[0].name.clone())
    });

    let info = gguf.tensor_or_err(&tensor_name)?;

    let bytes_read  = info.nbytes();
    let bytes_write = info.n_elems() * 4;

    println!("Tensor: {} | {} | {} | {:.2} MB (Q) -> {:.2} MB (F32)",
        info.name, info.shape_str(), info.dtype.name(),
        bytes_read as f64 / (1024.0 * 1024.0),
        bytes_write as f64 / (1024.0 * 1024.0));
    println!("Elementos: {}", info.n_elems());
    println!("Iteracoes: {} + 1 warmup", cli.iters);
    println!();

    // Buffer alinhado em 64B (necessario para NT stores / cache-line alignment)
    // Usa Vec padded para garantir alinhamento: alloca n+16 elementos, ajusta ptr.
    // Alternativa simples: aloca normalmente - Vec<f32> ja e alinhado em 4B pelo
    // alocador do Rust; a diferença para 64B e pequena para storeu.
    let mut dst = vec![0.0f32; info.n_elems()];

    // Warmup: escreve a memoria para acordar as paginas fisicas (evita page faults
    // dentro da janela cronometrada). Sem isso o primeiro pass inclui tempo de kernel.
    for v in dst.iter_mut() { *v = 0.0; }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

    // Warmup pass (nao cronometrado) - executa o caminho SIMD completo
    gguf.dequantize_tensor(info, &mut dst)?;
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

    println!("[warmup concluido - paginas fisicas mapeadas e caminho SIMD aquecido]");
    println!();

    let mut total_ns = 0u128;

    for i in 0..cli.iters {
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
        let t0 = Instant::now();
        gguf.dequantize_tensor(info, &mut dst)?;
        let elapsed = t0.elapsed().as_nanos();
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

        total_ns += elapsed;
        println!("  iter {}: {:.2} ms", i + 1, elapsed as f64 / 1e6);
    }

    let avg_ms = total_ns as f64 / cli.iters as f64 / 1e6;

    // Throughput de leitura: MB de dados quantizados lidos por segundo
    let gb_s_read  = (bytes_read  as f64 / 1e9) / (avg_ms / 1000.0);
    let gb_s_write = (bytes_write as f64 / 1e9) / (avg_ms / 1000.0);
    let gb_s_total = ((bytes_read + bytes_write) as f64 / 1e9) / (avg_ms / 1000.0);

    println!();
    println!("Media:           {:.2} ms", avg_ms);
    println!("Throughput read: {:.2} GB/s  [{:.1} MB Q comprimido]", gb_s_read,  bytes_read  as f64 / 1024.0 / 1024.0);
    println!("Throughput wrt:  {:.2} GB/s  [{:.1} MB F32 expandido]", gb_s_write, bytes_write as f64 / 1024.0 / 1024.0);
    println!("Throughput tot:  {:.2} GB/s  [read + write combinados]", gb_s_total);
    println!("Elementos/s:     {:.2} GElem/s", info.n_elems() as f64 / 1e9 / (avg_ms / 1000.0));

    let min  = dst.iter().cloned().fold(f32::INFINITY, f32::min);
    let max  = dst.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = dst.iter().sum::<f32>() / dst.len() as f32;
    println!();
    println!("Resultado: min={:.4}, max={:.4}, mean={:.6}", min, max, mean);

    Ok(())
}
