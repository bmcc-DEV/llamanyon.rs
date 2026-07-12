use std::time::Instant;
use clap::Parser;
use anyhow::Result;
use swamp_engine::executor::{InferenceRequest, ModelExecutor};
use swamp_engine::Model;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "swamp-benchmark-prefill")]
struct Args {
    model: String,
    tokenizer: String,
    #[arg(long, default_value = "128")]
    prompt_tokens: usize,
    #[arg(long, default_value = "1")]
    repeats: usize,
    #[arg(long, default_value = "6")]
    n_threads: usize,
    #[arg(long)]
    qat: bool,
    #[arg(long, default_value = "1")]
    pipeline: usize,
}

fn dummy_tokens(n: usize) -> Vec<usize> {
    (0..n).map(|i| 1usize + (i % 100)).collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    swamp_engine::tokenizer::init_tokenizer(&args.tokenizer);

    println!("Loading model...");
    let model = Arc::new(Model::load(&args.model)?);
    model.print_info();

    // Init sensitivity map if --qat
    if args.qat {
        let num_layers = model.config.num_layers;
        swamp_engine::linear::init_sensitivity(&model.gguf, num_layers);
    }

    let executor = ModelExecutor::new(model.clone());

    println!("Benchmark: {} prompt tokens, {} repeats, {} threads, pipeline={}",
        args.prompt_tokens, args.repeats, args.n_threads, args.pipeline);

    for rep in 0..args.repeats {
        let t0 = Instant::now();

        if args.pipeline > 1 {
            // Pipeline concurrente: N requests em paralelo
            let requests: Vec<InferenceRequest> = (0..args.pipeline).map(|_| {
                InferenceRequest {
                    prompt: Some("What is the meaning of life?".into()),
                    messages: None,
                    max_tokens: 10,
                    temperature: 0.0,
                    top_k: 1,
                    top_p: 1.0,
                }
            }).collect();

            let results = executor.generate_batch(requests, args.pipeline);
            let total_elapsed = t0.elapsed().as_secs_f64();
            let ok_count = results.iter().filter(|r| r.is_ok()).count();
            println!("--- Repeat {} (pipeline={}) ---", rep + 1, args.pipeline);
            println!("  Total: {:.1}s for {} requests ({:.1} req/s)",
                total_elapsed, args.pipeline, args.pipeline as f64 / total_elapsed);
            println!("  OK: {}/{}", ok_count, args.pipeline);
        } else {
            // Single request via executor (uses GPU GEMV + CUDA Graphs + attention sparsa)
            let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);

            // Prefill only benchmark
            let req = InferenceRequest {
                prompt: Some(" ".repeat(args.prompt_tokens)),
                messages: None,
                max_tokens: 0,
                temperature: 0.0,
                top_k: 1,
                top_p: 1.0,
            };

            let mut exec = executor.clone();
            let handle = std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async move {
                    let _ = exec.generate(req, tx).await;
                });
            });

            // Collect output
            let mut output = String::new();
            while let Some(msg) = rx.blocking_recv() {
                output.push_str(&msg);
            }
            handle.join().unwrap();

            let total_elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            println!("--- Repeat {} ---", rep + 1);
            println!("  Total: {:.1}ms", total_elapsed);

            // Decode benchmark: generate 10 tokens
            let (tx2, mut rx2) = tokio::sync::mpsc::channel::<String>(64);
            let req2 = InferenceRequest {
                prompt: None,
                messages: None,
                max_tokens: 10,
                temperature: 0.0,
                top_k: 1,
                top_p: 1.0,
            };

            let t1 = Instant::now();
            let mut exec2 = executor.clone();
            let handle2 = std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async move {
                    let _ = exec2.generate(req2, tx2).await;
                });
            });

            while let Some(msg) = rx2.blocking_recv() {
                // consume output
            }
            handle2.join().unwrap();

            let decode_ms = t1.elapsed().as_secs_f64() * 1000.0;
            let per_token = decode_ms / 10.0;
            println!("  Decode: {:.1}ms for 10 tokens ({:.1} ms/tok, {:.0} tok/s)",
                decode_ms, per_token, 10.0 / (decode_ms / 1000.0));
        }
    }
    Ok(())
}
