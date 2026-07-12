// swamp-engine/src/pipeline.rs
// Pipeline — grafo de estágios conectando modelos para execução em cadeia.
//
// Cada estágio usa um modelo registrado no ModelRegistry e produz saída
// que alimenta o próximo estágio. A pipeline roda N estágios em paralelo
// (pipeline parallelism) quando possível.
//
// Para 7 modelos (4 GPU + 3 CPU):
//   1. Qwen2.5-Omni-7B  → interpreta intent do usuário, tool calling
//   2. DeepSeek-R1-1.5B → raciocínio lógico, plano de execução
//   3. GLM-4.7-3B       → geração de código/scripts
//   4. Phi-4-mini-3.8B  → auditoria de segurança, verificação
//   5. SmolVLM-500M     → limpeza de logs (CPU)
//   6. Llama 4 1B       → envelopamento A2A (CPU)
//   7. GLM-4.5-Air      → relatório final (CPU)

use crate::model_registry::ModelRegistry;
use std::sync::Arc;

/// Tipo de entrada/saída de um estágio
#[derive(Debug, Clone)]
pub enum StageIO {
    Text(String),
    Tokens(Vec<usize>),
    Structured(String), // JSON/YAML
}

/// Um estágio da pipeline
#[derive(Clone)]
pub struct Stage {
    pub name: String,
    pub model_name: String,
    pub system_prompt: String,
    pub temperature: f32,
    pub max_tokens: usize,
}

/// Resultado de um estágio
#[derive(Clone)]
pub struct StageResult {
    pub stage_name: String,
    pub output: StageIO,
    pub tokens_generated: usize,
    pub elapsed_ms: f64,
}

/// Configuração completa da pipeline
pub struct PipelineConfig {
    pub stages: Vec<Stage>,
}

impl PipelineConfig {
    /// Pipeline padrão para os 7 modelos descritos
    pub fn default_7model() -> Self {
        Self {
            stages: vec![
                Stage {
                    name: "intent".into(),
                    model_name: "qwen-omni-7b".into(),
                    system_prompt: "You are the human interface. Parse the user request into a structured tool call.".into(),
                    temperature: 0.1,
                    max_tokens: 256,
                },
                Stage {
                    name: "strategy".into(),
                    model_name: "deepseek-r1-1.5b".into(),
                    system_prompt: "You are the strategy director. Create a logical execution plan.".into(),
                    temperature: 0.3,
                    max_tokens: 512,
                },
                Stage {
                    name: "codegen".into(),
                    model_name: "glm-flash-3b".into(),
                    system_prompt: "You are the software engineer. Generate precise scripts.".into(),
                    temperature: 0.2,
                    max_tokens: 1024,
                },
                Stage {
                    name: "audit".into(),
                    model_name: "phi-4-mini-3.8b".into(),
                    system_prompt: "You are the safety auditor. Check for infinite loops, destructuve commands, and policy violations.".into(),
                    temperature: 0.1,
                    max_tokens: 256,
                },
                Stage {
                    name: "cleanup".into(),
                    model_name: "smol-vlm".into(),
                    system_prompt: "Clean terminal logs into structured data.".into(),
                    temperature: 0.1,
                    max_tokens: 128,
                },
                Stage {
                    name: "envelope".into(),
                    model_name: "llama4-1b".into(),
                    system_prompt: "Envelope the cleaned data into strict A2A protocol format.".into(),
                    temperature: 0.0,
                    max_tokens: 256,
                },
                Stage {
                    name: "report".into(),
                    model_name: "glm-air".into(),
                    system_prompt: "Write a human-readable technical report from the audit results.".into(),
                    temperature: 0.4,
                    max_tokens: 512,
                },
            ],
        }
    }
}

/// Pipeline executor: roda cada estágio sequencialmente, alimentando saída
/// de um como entrada do próximo.
pub struct PipelineExecutor {
    config: PipelineConfig,
}

impl PipelineExecutor {
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Roda a pipeline completa para uma requisição.
    /// `model_getter`: função que retorna um modelo do registry.
    /// `run_stage_fn`: função que executa inferência em um modelo e retorna
    ///   o texto gerado.
    pub fn execute(
        &self,
        user_input: &str,
        model_getter: impl Fn(&str) -> Option<Arc<crate::model::Model>>,
        run_stage_fn: impl Fn(&Stage, String, &crate::model::Model) -> anyhow::Result<(String, usize, std::time::Duration)>,
    ) -> anyhow::Result<Vec<StageResult>> {
        let mut results = Vec::with_capacity(self.config.stages.len());
        let mut current_input = user_input.to_string();

        for stage in &self.config.stages {
            let t0 = std::time::Instant::now();
            let model = model_getter(&stage.model_name)
                .ok_or_else(|| anyhow::anyhow!("Modelo '{}' não encontrado no registry", stage.model_name))?;

            let (output, tokens_gen, elapsed) = run_stage_fn(stage, current_input, &model)?;

            results.push(StageResult {
                stage_name: stage.name.clone(),
                output: StageIO::Text(output.clone()),
                tokens_generated: tokens_gen,
                elapsed_ms: elapsed.as_secs_f64() * 1000.0,
            });

            current_input = output;
        }

        Ok(results)
    }

    /// Roda N requisições em paralelo via pipeline parallelism.
    /// `execute_concurrent` — cada estágio roda em sua própria thread,
    /// conectado por canais mpsc. Enquanto GPU processa "codegen" da req i,
    /// CPU já faz "cleanup" da req i-1.
    ///
    /// `requests`: N inputs de usuário
    /// `max_concurrency`: AIMD budget — quantas reqs ativas simultaneamente
    pub fn execute_concurrent(
        &self,
        requests: Vec<String>,
        max_concurrency: usize,
        model_getter: Arc<dyn Fn(&str) -> Option<Arc<crate::model::Model>> + Send + Sync>,
        run_stage: Arc<dyn Fn(&Stage, String, Arc<crate::model::Model>) -> anyhow::Result<(String, usize, std::time::Duration)> + Send + Sync>,
    ) -> Vec<anyhow::Result<Vec<StageResult>>> {
        use std::sync::mpsc;
        use std::thread;

        let n_stages = self.config.stages.len();
        let n_requests = requests.len();
        if n_requests == 0 || n_stages == 0 { return Vec::new(); }

        // Flow control via bounded sync_channel
        let mut channels: Vec<mpsc::SyncSender<(usize, String)>> = Vec::with_capacity(n_stages);
        let mut receivers: Vec<mpsc::Receiver<(usize, String)>> = Vec::with_capacity(n_stages);
        for _ in 0..n_stages {
            let (tx, rx) = mpsc::sync_channel::<(usize, String)>(max_concurrency);
            channels.push(tx);
            receivers.push(rx);
        }

        // Shared results: Arc<Mutex<Vec<Option<Vec<StageResult>>>>>
        let results: Arc<std::sync::Mutex<Vec<Option<Vec<StageResult>>>>> =
            Arc::new(std::sync::Mutex::new(vec![None; n_requests]));

        // Flow control counter: how many requests are in the pipeline
        let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Spawn stage threads
        let mut handles = Vec::with_capacity(n_stages);
        for stage_idx in 0..n_stages {
            let stage = self.config.stages[stage_idx].clone();
            let rx = receivers.remove(0);
            let next_tx = if stage_idx + 1 < n_stages {
                Some(channels[stage_idx + 1].clone())
            } else {
                None
            };
            let mg = model_getter.clone();
            let rf = run_stage.clone();
            let res = results.clone();
            let flow = in_flight.clone();

            handles.push(thread::spawn(move || {
                while let Ok((req_idx, input)) = rx.recv() {
                    let t0 = std::time::Instant::now();
                    let model = match (mg)(&stage.model_name) {
                        Some(m) => m,
                        None => continue,
                    };
                    let result = (rf)(&stage, input, model);
                    match result {
                        Ok((output, tokens_gen, elapsed)) => {
                            // Store result
                            {
                                let mut r = res.lock().unwrap();
                                if let Some(ref mut slot) = r[req_idx] {
                                    slot.push(StageResult {
                                        stage_name: stage.name.clone(),
                                        output: StageIO::Text(output.clone()),
                                        tokens_generated: tokens_gen,
                                        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
                                    });
                                } else {
                                    r[req_idx] = Some(vec![StageResult {
                                        stage_name: stage.name.clone(),
                                        output: StageIO::Text(output.clone()),
                                        tokens_generated: tokens_gen,
                                        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
                                    }]);
                                }
                            }
                            // Send to next stage
                            if let Some(ref tx) = next_tx {
                                let _ = tx.send((req_idx, output));
                            }
                        }
                        Err(_) => {}
                    }
                }
            }));
        }

        // Send all requests into stage 0 (sync_channel blocks at max_concurrency)
        for (req_idx, input) in requests.into_iter().enumerate() {
            let _ = channels[0].send((req_idx, input));
        }

        // Drop senders so receivers stop
        drop(channels);
        for h in handles {
            let _ = h.join();
        }

        let final_results = results.lock().unwrap();
        final_results.iter().map(|r| {
            match r {
                Some(stages) => Ok(stages.clone()),
                None => Err(anyhow::anyhow!("request failed")),
            }
        }).collect()
    }
}
