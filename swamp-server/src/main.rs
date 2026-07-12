// swamp-server/src/main.rs
// Servidor HTTP de inferencia para LLamañón.rs usando Axum, Tokio e LuaJIT

mod circuit_breaker;
mod metrics;
mod scheduler;

use axum::{
    routing::{get, post},
    Router,
    extract::{State, Json},
    response::sse::{Event, Sse},
    response::IntoResponse,
    http::StatusCode,
};
use clap::Parser;
use std::sync::Arc;
use std::time::Duration;
use std::convert::Infallible;
use tokio::sync::mpsc;
use futures::stream::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

use swamp_engine::{Model, ModelExecutor};
use circuit_breaker::CircuitBreaker;
use metrics::ConcurrencyMetrics;
use scheduler::{PriorityQueueScheduler, QueuedRequest};

#[derive(Parser)]
#[command(name = "swamp-server")]
#[command(about = "Swamp - Servidor HTTP de Inferencia")]
struct Args {
    /// Caminho do modelo GGUF
    #[arg(short, long, default_value = "/media/bruno/3e94d163-2a59-473e-bcc5-09148350a987/MODELS/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")]
    model: String,

    /// Porta do servidor
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Tamanho maximo do batch para o Continuous Batcher
    #[arg(short = 'b', long, default_value_t = 4)]
    batch_size: usize,
}

// Estado global do servidor
struct AppState {
    scheduler: Arc<PriorityQueueScheduler>,
    circuit_breaker: CircuitBreaker,
    metrics: ConcurrencyMetrics,
}

#[derive(serde::Deserialize)]
struct ApiRequest {
    prompt: Option<String>,
    messages: Option<Vec<swamp_engine::chat_template::ChatMessage>>,
    #[serde(default = "default_user_tier")]
    user_tier: String,
    #[serde(default = "default_workload_type")]
    workload_type: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_top_p")]
    top_p: f32,
}

fn default_user_tier() -> String { "FREE".to_string() }
fn default_workload_type() -> String { "CHAT".to_string() }
fn default_max_tokens() -> usize { 100 }
fn default_temperature() -> f32 { 0.7 }
fn default_top_k() -> usize { 40 }
fn default_top_p() -> f32 { 0.95 }

// Wrapper para converter o Receiver em um Stream do Axum SSE
struct ReceiverStream {
    rx: mpsc::Receiver<String>,
}

impl Stream for ReceiverStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx).map(|opt| opt.map(|s| {
            Ok(Event::default().data(s.replace("\n", "\\n").replace("\r", "")))
        }))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Inicializa logs
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    tracing::info!("=== Inicializando LLamañón.rs Server ===");
    tracing::info!("Carregando modelo GGUF de: {}", args.model);

    // Carrega o modelo
    let model = Arc::new(Model::load(&args.model)?);
    model.print_info();

    // Inicializa infraestrutura
    let policy_path = "policies/swamp_policy.lua";
    let scheduler = Arc::new(PriorityQueueScheduler::new(policy_path, args.batch_size)?);
    let circuit_breaker = CircuitBreaker::new(5, Duration::from_secs(30));
    let metrics = ConcurrencyMetrics::new();

    // Inicializa o executor e o loop em background do batcher
    let executor = ModelExecutor::new(model.clone());
    let scheduler_clone = scheduler.clone();
    let metrics_clone = metrics.clone();

    // Inicializa Tokenizer
    swamp_engine::tokenizer::init_tokenizer("/media/bruno/3e94d163-2a59-473e-bcc5-09148350a987/MODELS/tinyllama_tokenizer.json");

    tokio::spawn(async move {
        scheduler_clone.start_batcher_loop(executor, metrics_clone).await;
    });

    let state = Arc::new(AppState {
        scheduler,
        circuit_breaker,
        metrics,
    });

    // Roteamento
    let app = Router::new()
        .route("/generate", post(handle_generate))
        .route("/generate-stream", post(handle_generate_stream))
        .route("/metrics", get(handle_metrics))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", args.port)).await?;
    tracing::info!("Servidor HTTP rodando em http://localhost:{}", args.port);
    axum::serve(listener, app).await?;

    Ok(())
}

// Endpoint /generate: responde após a conclusão de todos os tokens
async fn handle_generate(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ApiRequest>,
) -> impl IntoResponse {
    // Valida Circuit Breaker
    if state.circuit_breaker.is_open() {
        state.metrics.rejected_requests.inc();
        return (StatusCode::SERVICE_UNAVAILABLE, "Serviço sobrecarregado. CircuitBreaker aberto.").into_response();
    }

    // Limite de backpressure (maximo 20 requests na fila)
    if state.scheduler.len().await > 20 {
        state.metrics.rejected_requests.inc();
        return (StatusCode::TOO_MANY_REQUESTS, "Fila de processamento cheia. Tente novamente mais tarde.").into_response();
    }

    let (tx, mut rx) = mpsc::channel(body.max_tokens + 1);

    let queued = QueuedRequest {
        id: rand::random::<u64>(),
        prompt: body.prompt,
        messages: body.messages,
        user_tier: body.user_tier,
        workload_type: body.workload_type,
        max_tokens: body.max_tokens,
        temperature: body.temperature,
        top_k: body.top_k,
        top_p: body.top_p,
    };

    if let Err(e) = state.scheduler.submit(queued, tx).await {
        state.circuit_breaker.record_failure();
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("Erro ao enfileirar requisição: {:?}", e)).into_response();
    }

    // Coleta todos os tokens
    let mut response_text = String::new();
    while let Some(token) = rx.recv().await {
        response_text.push_str(&token);
    }

    state.circuit_breaker.record_success();
    (StatusCode::OK, response_text).into_response()
}

// Endpoint /generate-stream: envia tokens em tempo real usando Server-Sent Events (SSE)
async fn handle_generate_stream(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ApiRequest>,
) -> impl IntoResponse {
    // Valida Circuit Breaker
    if state.circuit_breaker.is_open() {
        state.metrics.rejected_requests.inc();
        return (StatusCode::SERVICE_UNAVAILABLE, "Serviço sobrecarregado. CircuitBreaker aberto.").into_response();
    }

    // Limite de backpressure
    if state.scheduler.len().await > 20 {
        state.metrics.rejected_requests.inc();
        return (StatusCode::TOO_MANY_REQUESTS, "Fila cheia.").into_response();
    }

    let (tx, rx) = mpsc::channel(body.max_tokens + 1);

    let queued = QueuedRequest {
        id: rand::random::<u64>(),
        prompt: body.prompt,
        messages: body.messages,
        user_tier: body.user_tier,
        workload_type: body.workload_type,
        max_tokens: body.max_tokens,
        temperature: body.temperature,
        top_k: body.top_k,
        top_p: body.top_p,
    };

    if let Err(e) = state.scheduler.submit(queued, tx).await {
        state.circuit_breaker.record_failure();
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("Erro: {:?}", e)).into_response();
    }

    state.circuit_breaker.record_success();

    let stream = ReceiverStream { rx };
    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::new())
        .into_response()
}

// Endpoint /metrics para exportação Prometheus
async fn handle_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    let mut buffer = Vec::new();
    let metric_families = state.metrics.registry.gather();
    encoder.encode(&metric_families, &mut buffer).unwrap();

    let text = String::from_utf8(buffer).unwrap_or_default();
    (StatusCode::OK, [("content-type", "text/plain; version=0.0.4")], text)
}
