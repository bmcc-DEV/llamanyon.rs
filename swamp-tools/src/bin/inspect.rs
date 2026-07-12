// swamp-tools/src/bin/inspect.rs
// swamp-inspect: inspeciona um arquivo GGUF e exibe metadados e tensores

use anyhow::Result;
use clap::Parser;
use swamp_gguf::GgufFile;

#[derive(Parser)]
#[command(name = "swamp-inspect")]
#[command(about = "Swamp - inspeciona modelos GGUF")]
struct Cli {
    /// Caminho para o arquivo .gguf
    model: String,

    /// Mostra lista de tensores
    #[arg(short, long)]
    tensors: bool,

    /// Mostra apenas tensores cujo nome contenha este padrao
    #[arg(short, long)]
    filter: Option<String>,

    /// Numero maximo de tensores a exibir (0 = todos)
    #[arg(short = 'n', long, default_value = "50")]
    max: usize,

    /// Mostra metadados detalhados
    #[arg(short, long)]
    metadata: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let gguf = GgufFile::open(&cli.model)?;

    // Cabecalho
    println!("=== Swamp - swamp-inspect ===");
    println!("Arquivo:       {}", cli.model);
    println!("Arquitetura:   {}", gguf.architecture());
    println!("Tensores:      {}", gguf.n_tensors());
    println!("Tamanho mmap:  {} MB", gguf.file_size() / (1024 * 1024));
    println!();

    // Metadados de arquitetura
    let md = &gguf.metadata;
    println!("--- Metadados de Modelo ---");

    if let Ok(v) = md.get_u64("general.parameter_count") {
        println!("  Parametros:    {:.2}B", v as f64 / 1e9);
    }
    if let Some(v) = md.get("general.name").and_then(|v| v.as_str()) {
        println!("  Nome:          {}", v);
    }
    if let Some(v) = md.get("general.base_model.0.name").and_then(|v| v.as_str()) {
        println!("  Base:          {}", v);
    }

    if let Ok(nl) = md.n_layer() {
        println!("  Camadas:       {}", nl);
    }
    if let Ok(ne) = md.n_embd() {
        println!("  Embedding:     {}", ne);
    }
    if let Ok(nh) = md.n_head() {
        println!("  Atencao heads: {} (KV: {})", nh, md.n_kv_head());
    }
    println!("  Contexto:      {}", md.n_ctx_train());
    println!("  RoPE base:     {}", md.rope_freq_base());
    println!("  RoPE dim:      {}", md.rope_dimension());

    if let Some(v) = md.get("tokenizer.ggml.model").and_then(|v| v.as_str()) {
        println!("  Tokenizer:     {}", v);
    }
    if let Some(v) = md.get("tokenizer.ggml.tokens").and_then(|v| v.as_string_array()) {
        println!("  Vocab size:    {}", v.len());
    }

    // Metadados completos se solicitado
    if cli.metadata {
        println!();
        println!("--- Todos os Metadados ({}) ---", md.0.len());
        let mut keys: Vec<&String> = md.0.keys().collect();
        keys.sort();
        for k in keys {
            let val = &md.0[k];
            match val {
                swamp_gguf::MetadataValue::String(s) => println!("  {} = {:?}", k, s),
                swamp_gguf::MetadataValue::U32(v)    => println!("  {} = {}", k, v),
                swamp_gguf::MetadataValue::U64(v)    => println!("  {} = {}", k, v),
                swamp_gguf::MetadataValue::F32(v)    => println!("  {} = {}", k, v),
                swamp_gguf::MetadataValue::Bool(v)   => println!("  {} = {}", k, v),
                swamp_gguf::MetadataValue::Array(_)  => println!("  {} = [array]", k),
                v => println!("  {} = {:?}", k, v),
            }
        }
    }

    // Tensores
    if cli.tensors || cli.filter.is_some() {
        println!();
        println!("--- Tensores ---");

        let filtered: Vec<_> = gguf.tensors.iter()
            .filter(|t| {
                cli.filter.as_ref()
                    .map(|f| t.name.contains(f.as_str()))
                    .unwrap_or(true)
            })
            .collect();

        let total = filtered.len();
        let shown = if cli.max == 0 { total } else { total.min(cli.max) };

        // Calcula tamanho total do modelo
        let total_bytes: usize = gguf.tensors.iter().map(|t| t.nbytes()).sum();
        println!("  Total filtrado: {} / {} tensores", total, gguf.n_tensors());
        println!("  Tamanho total:  {:.2} GB", total_bytes as f64 / 1e9);
        println!();
        println!("  {:<50} {:>12} {:>8} {:>10}", "Nome", "Shape", "DType", "Bytes");
        println!("  {}", "-".repeat(86));

        for t in filtered.iter().take(shown) {
            println!("  {:<50} {:>12} {:>8} {:>10}",
                &t.name,
                t.shape_str(),
                t.dtype.name(),
                t.nbytes()
            );
        }

        if shown < total {
            println!("  ... e mais {} tensores (use -n 0 para ver todos)", total - shown);
        }

        // Distribuicao de tipos de quantizacao
        println!();
        println!("  --- Distribuicao de DTypes ---");
        let mut dtype_counts: std::collections::HashMap<String, (usize, usize)> = std::collections::HashMap::new();
        for t in &gguf.tensors {
            let e = dtype_counts.entry(t.dtype.name().to_string()).or_default();
            e.0 += 1;
            e.1 += t.nbytes();
        }
        let mut dtype_list: Vec<_> = dtype_counts.iter().collect();
        dtype_list.sort_by(|a, b| b.1.1.cmp(&a.1.1));
        for (dtype, (count, bytes)) in dtype_list {
            println!("  {:>10}: {:>4} tensores, {:.2} GB",
                dtype, count, *bytes as f64 / 1e9);
        }
    }

    println!();
    println!("OK - arquivo GGUF valido");
    Ok(())
}
