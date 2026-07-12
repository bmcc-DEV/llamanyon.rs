# Virtualização de Clock, Threads, Cache e NPU no LLLamañón.rs / SwampLLM

**Status:** parcialmente implementado. Este documento junta a estratégia discutida com o estado real do código em `swamp-engine/` — o que já existe, o que foi corrigido, e o que ainda é conceito.

**Hardware-alvo:** i5-11260H (Tiger Lake, sem NPU — NPU só chegou no Meteor Lake), GTX 1650 4GB (Turing sem tensor cores, TU117), 24GB DDR4, NVMe.

---

## 0. O problema real

Esse hardware não tem NPU dedicada e não tem VRAM suficiente pra manter um modelo grande + KV cache inteiramente na GPU. As saídas óbvias (quantizar mais, usar modelo menor) já estão sendo exploradas em `swamp-kernels` (Q4_K, Q6_K, Q8_0). A pergunta que sobra é arquitetural: **como tratar CPU, GPU, RAM e NVMe como um único domínio heterogêneo de tempo, execução e memória**, em vez de quatro subsistemas isolados que se falam por heurísticas estáticas.

Não é sobre inventar hardware. É sobre construir a camada de software que várias peças de pesquisa recente (2024-2026) já validam separadamente — PagedAttention, offload de KV pra NVMe, execução híbrida CPU/GPU tipo PowerInfer, drivers de NPU abertos (AMD XDNA, Intel NPU) — mas que ninguém combinou nos três eixos (tempo, execução, memória) num único escalonador adaptativo pra hardware de consumidor.

---

## 1. As quatro camadas

### 1.1 Clock virtual — `swamp-engine/src/hlc.rs`

TSC da CPU, eventos de stream CUDA, e completions de I/O do NVMe vivem em domínios de tempo diferentes. Pra correlacionar decisões entre eles (quando prefetch, quando trocar de camada, quanto tempo um GEMV realmente levou) é preciso um referencial único.

**Implementado:** `WallClock` (TSC calibrado contra `CLOCK_MONOTONIC`) + `HybridClock` (HLC clássico: componente wall + contador lógico, com `witness()` pra incorporar timestamps externos como eventos de GPU). `ProfileSink` usa isso pra cronometrar cada camada do transformer.

**Bug encontrado e corrigido:** `WallClock::measure_scale` tinha um erro de precedência de operador (`<< 32 / elapsed` em vez de `(<< 32) / elapsed`) que fazia o clock reportar tempo ~170× mais lento que o real, corrompendo silenciosamente todo profiling. Corrigido.

### 1.2 Threads virtuais — scheduler M:N

A ideia original: tratar workers de CPU (AVX2/AVX-512), streams CUDA e filas de I/O como "green threads" de um único executor work-stealing, decidindo em tempo real onde rodar cada kernel — o princípio do PowerInfer (neurônios quentes na GPU, frios na CPU) aplicado no nível de scheduler, não só de neurônio.

**Implementado, parcialmente:**
- `swamp-engine/src/scheduler.rs`: gerenciamento de streams CUDA por camada, com CUDA Graphs cacheados por `seq_len` e fallback pra CPU quando a GPU falha. É M:N no sentido GPU-stream ↔ camada, não um executor unificado genérico.
- `swamp-engine/src/vnpu.rs` (novo): escalonador adaptativo especificamente pro particionamento de threads CPU num GEMV — ver seção 1.4, é a peça mais próxima da ideia de "decidir em tempo real, não estaticamente".

**Não implementado:** um executor único que trate streams CUDA e filas io_uring como cidadãos de primeira classe ao lado de threads CPU. Hoje são três mecanismos paralelos (rayon pool, streams CUDA em `scheduler.rs`, `madvise` síncrono em `prefetch.rs`), não um só.

### 1.3 Cache de LM virtualizado — `swamp-engine/src/cache.rs` + `prefetch.rs`

Hierarquia de três níveis: VRAM (quente) / RAM (morna, paginada) / NVMe (fria). Contexto longo em GPU de 4GB é fisicamente impossível sem isso.

**Implementado:**
- `PagedKVCache`: paginação real com LRU intrusivo (prev/next por página), eviction pra arquivo cold (`/tmp/swamp_cache`), reload sob demanda. Testes cobrindo eviction e reload.
- `PrefetchEngine`: `mmap` + `madvise(MADV_WILLNEED)` pros **pesos do modelo** (não o KV cache) — streaming de pesos do NVMe sem carregar o `.gguf` inteiro, na linha do "LLM in a Flash".

**Bug encontrado e corrigido:** o pre-heat de páginas frias (`ensure_pages_hot`) rodava incondicionalmente antes de saber se o caminho GPU ia ter sucesso. Quando a GPU está ativa (lendo do seu próprio buffer residente `d_k_buf`/`d_v_buf`, não do `PagedKVCache`), esse pre-heat forçava recarregar todo o histórico frio de volta pra RAM a cada token — derrotando o próprio propósito do eviction exatamente no cenário em que mais importa (GPU ativa + contexto longo). Corrigido: o pre-heat só roda no fallback CPU.

**Não implementado:** promoção de blocos de KV pra VRAM como tier explícito gerenciado pelo mesmo scheduler que gerencia RAM/NVMe (hoje a GPU tem seu próprio buffer separado, alocado por camada, não integrado à hierarquia de páginas).

### 1.4 NPU virtual — `swamp-engine/src/vnpu.rs`

A CPU não tem NPU (11ª geração). A ideia: construir uma camada que expõe uma interface parecida com NPU pro resto do software, mas roteia pra CPU/GPU por trás — inspirada na arquitetura real de drivers abertos (AMD `amdxdna`, Intel NPU driver): SHIM em userspace + submissão de comando via fila + IOCTL, gerenciamento de buffer via GEM. Não é clonar silício — é estudar a interface de um driver open-source real e reimplementar o contrato, tipo Wine reimplementando Win32.

**Implementado (versão inicial, escopo real, não o driver completo):**
`VirtualNpuScheduler` — não é um character device nem um shim de IOCTL ainda (isso exigiria kernel module / VFIO, root, fora do escopo de uma primeira versão em espaço de usuário puro). É a peça de decisão que um shim desses precisaria por trás: mede o custo real de cada GEMV via `HybridClock` (EWMA de ns/linha, ponto fixo Q32, sem lock), e decide quantas threads CPU usar como um controlador de malha fechada — se a partição escolhida não entrega o orçamento de latência (2ms), a próxima chamada tende a pedir mais threads; se sobrar margem, pede menos. Isso substitui profiling offline estático (o problema conhecido do particionamento fixo do PowerInfer) por aprendizado online.

Backend GPU existe como variante reservada (`Backend::Gpu`) no enum, documentada como não implementada — hoje `swamp-gpu` só expõe `attention`, não GEMV/matmul. Quando existir, entra sem quebrar os call sites (eles só recebem `n_threads`/`backend` sugeridos).

**Wired em:** o GEMV mais caro do layer (FFN down-proj, `ffn_dim → embed_dim`) em `executor.rs`, usando o `HybridClock` já existente (`profiler.clock()`) como fonte de tempo — sem clock novo, sem subsistema paralelo.

**Bug relacionado encontrado e corrigido:** `n_threads` era capturado uma vez antes do `spawn_blocking` e nunca reatribuído — o hot-reload do `PolicyEngine` (Lua) trocava o script mas não tinha efeito nenhum no número de threads realmente em uso. Corrigido: `n_threads` agora é recomputado a cada reload (10 steps) via `local_policy.adapt_threads(...)`.

**Não implementado:** o shim de IOCTL/character device que faria software de terceiros (ONNX Runtime, OpenVINO) falar com isso pensando que é uma NPU real. `VirtualNpuScheduler` hoje é uma API interna do `swamp-engine`, não uma interface de driver exposta ao SO.

---

## 2. O que é genuinamente inédito aqui

Um escalonador único guiado por HLC que trata clock, execução e memória como uma coisa só — decide *quando* (clock), *onde* (CPU/GPU) e *o quê* (qual página buscar) com base em custo medido online, não em profiling offline. Cada peça isolada (PagedAttention, PowerInfer, drivers de NPU abertos) já existe publicada; a combinação dos três eixos num controlador adaptativo único, para hardware de consumidor sem NPU dedicada, é o que não tem equivalente direto na literatura hoje.

O `vnpu.rs` implementado é a primeira fatia real dessa ideia — pequena de propósito (só GEMV, só CPU), pra validar o padrão de controle antes de estender pros outros eixos.

---

## 3. Roadmap

Ordem sugerida, cada item depende do anterior:

1. **Estender `vnpu` pros outros GEMVs do layer** (q/k/v, gate/up) — hoje só o down-proj usa o scheduler adaptativo; os outros ainda usam `n_threads` estático.
2. **Mover `VirtualNpuScheduler` pro `ModelExecutor`** (vida mais longa que uma `generate()`) — aprende entre requests em vez de começar do zero (seed otimista) a cada chamada.
3. **Validar o handshake Rust↔Mojo** — `swamp-gpu/kernels/attention.mojo` existe mas não está no `build.rs`/`Makefile`; é bloqueador pra qualquer kernel novo em Mojo entrar no pipeline.
4. **GEMV na GPU** — só depois disso existir, `Backend::Gpu` em `vnpu.rs` ganha sentido, e o scheduler passa a decidir CPU-vs-GPU, não só quantas threads CPU.
5. **Tier de VRAM explícito no `PagedKVCache`** — hoje o buffer residente da GPU é separado do sistema de páginas; unificar sob o mesmo gerenciador permitiria o mesmo LRU/eviction decidir promoção pra VRAM, não só RAM↔NVMe.
6. **Shim de driver (IOCTL/character device)** — só faz sentido depois dos itens acima estarem estáveis; é a parte mais arriscada (kernel-level) e a que menos entrega valor imediato de performance.

---

## 4. Referências conceituais usadas como blueprint

- **PagedAttention / vLLM** — paginação de KV cache, base do `PagedKVCache`.
- **"LLM in a Flash" (Apple)** — streaming de pesos via `mmap`+`madvise`, base do `PrefetchEngine`.
- **PowerInfer** — execução híbrida CPU/GPU por neurônio quente/frio; motivou a substituição de particionamento estático por decisão online no `vnpu`.
- **AMD XDNA driver / Intel NPU Linux driver (open-source)** — blueprint de arquitetura (SHIM userspace + IOCTL + GEM) pro conceito de NPU virtual, ainda não implementado como shim real.
