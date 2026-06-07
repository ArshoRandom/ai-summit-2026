# vLLM Engine Options — for the *Scaling On-Premise AI* deck

This README documents the **vLLM engine arguments** that back the concepts in the
`OnPremAI.pptx` presentation ("Scaling On-Premise AI — From Ollama prototyping to vLLM
production"). Flags, choices, and defaults are taken from the **current stable** vLLM *Engine
Arguments* reference and grouped by the slide / idea each one relates to.

> Source: [vLLM (stable) — Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/).
> Engine args are organized into config groups (`ModelConfig`, `CacheConfig`, `ParallelConfig`,
> `SchedulerConfig`, `ObservabilityConfig`, …); the group is noted for each flag.

In stable vLLM these arguments are passed to `vllm serve` (online) or the `LLM` class (offline).
Boolean flags now come as a pair — e.g. `--enable-prefix-caching` / `--no-enable-prefix-caching`.
JSON-shaped configs accept either a JSON string or dotted keys, e.g.
`--structured-outputs-config.backend xgrammar`.

Only arguments with a clear tie to the talk are included. Unrelated arguments (LoRA serving,
multi-modal inputs, MoE expert-parallel internals, Mamba cache, etc.) are listed briefly at the end.

---

## What changed since older vLLM (and the deck's wording)

The deck and a lot of older blog posts reference v0.4-era flag names. On current vLLM:

- **`--gpu-memory-utilization` now defaults to `0.92`** (the deck says "90% = vLLM default" — that
  was true for v0.4.x; today the default is 0.92).
- **Prefix caching and chunked prefill are ON by default** in the V1 engine — you now *disable*
  them with `--no-enable-prefix-caching` / `--no-enable-chunked-prefill`.
- **`--guided-decoding-backend` → `--structured-outputs-config`** (a JSON config with a `backend` key).
- **All `--speculative-*` / `--ngram-*` flags → one `--speculative-config` JSON.**
- **`--max-seq-len-to-capture` is gone**; CUDA-graph capture is controlled by the compilation config
  (`--compilation-config` / `-O`) and the new `--performance-mode`.
- **`--rope-scaling` is gone** as a top-level flag; pass it through `--hf-overrides`.
- **`--disable-log-requests` / `--max-log-len` → `--enable-log-requests`** (off by default).

---

## How the deck maps to engine flags

| Deck slide / idea | Engine arguments |
|---|---|
| **vLLM speaks the OpenAI API** (slide 6, 16–18) | `--model`, `--max-model-len` |
| **PagedAttention — the "parking lot"** (slide 7) | `--gpu-memory-utilization`, `--block-size`, `--kv-cache-memory-bytes`, `--num-gpu-blocks-override` |
| **KV cache, the real cost** (slide 8) | `--kv-cache-dtype`, `--max-model-len`, `--cpu-offload-gb`, `--kv-offloading-size` |
| **Reusing work across users — prefix caching** (slide 9) | `--enable-prefix-caching` / `--no-enable-prefix-caching`, `--prefix-caching-hash-algo` |
| **Don't let one big request block everyone** (slide 10) | `--enable-chunked-prefill`, `--max-num-batched-tokens`, `--max-num-partial-prefills`, `--long-prefill-token-threshold`, `--performance-mode` |
| **Fairness / priority between teams** (slide 5) | `--scheduling-policy` |
| **Talking to other systems — structured output / tool use** (slide 11, 16) | `--structured-outputs-config` |
| **Split a big model across GPUs** (slide 12) | `--tensor-parallel-size`, `--pipeline-parallel-size`, `--distributed-executor-backend`, `--max-parallel-loading-workers`, `--disable-custom-all-reduce` |
| **Built-in metrics / dashboards** (slide 12) | `--disable-log-stats`, `--kv-cache-metrics`, `--enable-mfu-metrics`, `--otlp-traces-endpoint`, `--enable-log-requests` |
| **How many users per server — concurrency** (slide 14) | `--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization` |
| **Model & precision (Qwen BF16)** (slide 13–14) | `--model`, `--dtype`, `--quantization`, `--load-format`, `--trust-remote-code` |
| **Time-to-first-word / latency tuning** (slide 15) | `--enforce-eager`, `--performance-mode`, `-O` (compilation level), `--async-scheduling`, `--stream-interval` |
| **Long conversations / 32K context** (slide 14) | `--max-model-len`, `--disable-sliding-window`, `--hf-overrides` (RoPE scaling) |
| **"What's next" — speed via speculation** (slide 19) | `--speculative-config` |

---

## 1. Speaking the OpenAI API (slides 6, 16–18)

> *"Apps built for OpenAI work unchanged — just point them at your server instead."*
> Pointing Claude Code at your vLLM is exactly this: set `ANTHROPIC_BASE_URL` to the server and
> map the Opus/Sonnet/Haiku tiers to one served model.

| Flag (group) | Default | What it does |
|---|---|---|
| `--model` *(ModelConfig)* | `Qwen/Qwen3-0.6B` | Name or path of the HuggingFace model to serve (e.g. `Qwen/Qwen3.5-9B`). Also the name clients put in the `model` field, and the `model_name` value in metrics — it must match the deck's `ANTHROPIC_DEFAULT_*_MODEL` entries. |
| `--max-model-len` *(ModelConfig)* | from model config | Context length (prompt + output). Accepts human-readable values (`32k`, `25.6k`) and `-1`/`auto` to pick the largest length that fits in GPU memory. Caps how long a conversation can grow (slide 14's 2K → 32K rows). |
| `--tokenizer` *(ModelConfig)* | = model | Override the tokenizer path. |
| `--trust-remote-code` *(ModelConfig)* | `False` | Trust custom modeling code from the Hub — required by some current-gen open models (e.g. Qwen variants). |

```bash
vllm serve Qwen/Qwen3.5-9B \
  --max-model-len 32768
```

Then, as in slides 17–18, point Claude Code at it:

```jsonc
// ~/.claude/settings.json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_DEFAULT_OPUS_MODEL":   "Qwen/Qwen3.5-9B",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "Qwen/Qwen3.5-9B",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL":  "Qwen/Qwen3.5-9B"
  }
}
```

---

## 2. PagedAttention — the "parking lot" (slide 7)

> *"Memory broken into uniform slots, filled as needed… same GPU serves 3–5× more users."*

| Flag (group) | Default | What it does |
|---|---|---|
| `--gpu-memory-utilization` *(CacheConfig)* | `0.92` | Fraction of GPU memory (0–1) for this vLLM instance. The deck's operating points map here: **Conservative 0.80**, **Recommended 0.90**, **Aggressive 0.95**. Higher = bigger KV pool = more concurrent users, less spike headroom. (Per-instance; two instances on one GPU can each use 0.5.) |
| `--block-size` *(CacheConfig)* | model/platform default | Tokens per KV "parking slot" — the granularity of paged memory. |
| `--kv-cache-memory-bytes` *(CacheConfig)* | inferred | Pin the KV-cache size in bytes directly instead of deriving it from `--gpu-memory-utilization` (finer control; overrides the utilization fraction when set). |
| `--num-gpu-blocks-override` *(CacheConfig)* | — | Force a fixed number of KV blocks, ignoring profiling. Mainly for testing preemption. |

---

## 3. KV cache — the real cost (slide 8)

> *"The biggest consumer of GPU memory in production… memory, not compute, caps concurrent users."*

| Flag (group) | Default | What it does |
|---|---|---|
| `--kv-cache-dtype` *(CacheConfig)* | `auto` | Storage precision for the KV cache. `fp8` (`fp8_e4m3` / `fp8_e5m2`) roughly **halves** KV-cache memory vs FP16 — directly raising the concurrent-user numbers from slide 14. `auto` matches the model dtype. |
| `--max-model-len` *(ModelConfig)* | from config | Longer contexts grow the KV cache linearly (deck: "8× memory at 32K vs 4K") — capping this bounds per-request VRAM rent. |
| `--cpu-offload-gb` *(OffloadConfig)* | `0` | "Virtually" enlarge GPU memory by offloading weights to CPU RAM (needs fast CPU↔GPU interconnect). |
| `--kv-offloading-size` *(CacheConfig)* | — (off) | Enable CPU offloading of KV-cache blocks (GiB) under pressure, via `--kv-offloading-backend` (`native`/`lmcache`). |

---

## 4. Reusing work across users — prefix caching (slide 9)

> *"If 100 people ask questions about the same document, only process the document once… the first
> word appears 5–20× faster on requests that share a setup."*

| Flag (group) | Default | What it does |
|---|---|---|
| `--enable-prefix-caching` / `--no-enable-prefix-caching` *(CacheConfig)* | **on (V1)** | Automatic prefix caching — shared system prompts / instructions / documents are processed once and reused. The mechanism behind the deck's "70% cache reuse" and "~3× lower first-word latency" (slide 13). It's enabled by default now; use `--no-enable-prefix-caching` only to turn it off. |
| `--prefix-caching-hash-algo` *(CacheConfig)* | `sha256` | Hash used to match shared prefixes. `xxhash` is faster (non-cryptographic) — fine in trusted single-tenant deployments. |

---

## 5. Don't let one big request block everyone (slide 10)

> *"Take turns in small bites… Big request is slightly slower, but no one is stuck. Adjustable per
> workload: live chat → smaller bites, background → bigger bites."*

| Flag (group) | Default | What it does |
|---|---|---|
| `--enable-chunked-prefill` / `--no-enable-chunked-prefill` *(SchedulerConfig)* | **on (V1)** | Long prefills are **chunked** so other users' decode steps interleave between the pieces — the "small bites" scheduling on slide 10. |
| `--max-num-batched-tokens` *(SchedulerConfig)* | engine-computed | Tokens processed per iteration — the "bite size." Smaller → lower latency for chat; larger → higher throughput for batch jobs. |
| `--max-num-partial-prefills` *(SchedulerConfig)* | `1` | How many sequences may be partially prefilled at once under chunked prefill. |
| `--long-prefill-token-threshold` *(SchedulerConfig)* | `0` | A prompt longer than this counts as "long"; with `--max-long-partial-prefills` it lets short prompts jump ahead of long ones, improving latency. |
| `--performance-mode` *(VllmConfig)* | `balanced` | `interactivity` favors low per-request latency at small batch sizes; `throughput` favors aggregate tokens/sec at high concurrency — the deck's "live chat vs background processing" dial. |

---

## 6. Fairness & priority between teams (slide 5)

> The deck calls out Ollama's lack of "traffic rules — no way to give priority or set limits per team."

| Flag (group) | Default | What it does |
|---|---|---|
| `--scheduling-policy` *(SchedulerConfig)* | `fcfs` | `fcfs` = first-come-first-served; `priority` = serve by request priority (lower value first, arrival breaks ties) — the missing "fairness control" from slide 5. |

---

## 7. Talking to other systems — structured output & tool use (slides 11, 16)

> *"AI returns data in a predictable format every time… Constrained decoding makes open models
> reliable for agents."*

| Flag (group) | Default | What it does |
|---|---|---|
| `--structured-outputs-config` *(StructuredOutputsConfig)* | `backend='auto'` | Configures **constrained / guided decoding** (JSON-schema, regex, grammar) — what forces the structured `{"action": ...}` output on slide 11 and reliable tool-calling on slide 16. Set the engine via `--structured-outputs-config.backend` (e.g. `auto`, `xgrammar`, `guidance`, `outlines`). Replaces the old `--guided-decoding-backend`. |

```bash
vllm serve Qwen/Qwen3.5-9B --structured-outputs-config.backend xgrammar
```

---

## 8. Splitting a big model across GPUs (slide 12)

> *"Like a team of cooks each handling part of one dish. The big model is sliced up; the GPUs work
> together on each request."*

| Flag (group) | Default | What it does |
|---|---|---|
| `--tensor-parallel-size`, `-tp` *(ParallelConfig)* | `1` | Number of GPUs to **shard a single model across** (the "team of cooks"). Primary scale-up knob. |
| `--pipeline-parallel-size`, `-pp` *(ParallelConfig)* | `1` | Split the model by layers into pipeline stages across GPUs/nodes. |
| `--distributed-executor-backend` *(ParallelConfig)* | auto (`mp`/`ray`) | Backend for multi-GPU/multi-node serving. `mp` keeps it on one host; `ray` spans nodes. |
| `--max-parallel-loading-workers` *(ParallelConfig)* | — | Load the model in batches to avoid host-RAM OOM with tensor parallel + large models. |
| `--disable-custom-all-reduce` / `--no-` *(ParallelConfig)* | `False` | Fall back from vLLM's custom all-reduce kernel to NCCL — a multi-GPU compatibility knob. |

```bash
vllm serve <big-model> --tensor-parallel-size 4   # one model across 4 GPUs (slide 12)
```

---

## 9. Built-in metrics & dashboards (slide 12)

> *"How long users wait, how fast the first word appears, how full GPU memory is, queued vs running,
> how often cached prefixes are reused… if you can't see it, you can't fix it."*

vLLM exposes Prometheus metrics at `/metrics`. These flags control telemetry depth:

| Flag (group) | Default | What it does |
|---|---|---|
| `--disable-log-stats` *(EngineArgs)* | `False` (stats on) | Disables statistics logging. **Leave it on** to keep the slide-12 dashboard metrics. |
| `--kv-cache-metrics` / `--no-` *(ObservabilityConfig)* | `False` | Emits KV-cache residency metrics — lifetime, idle time, **reuse gaps** (the deck's "how often cached prefixes are reused"). Requires stats enabled. |
| `--enable-mfu-metrics` / `--no-` *(ObservabilityConfig)* | `False` | Model FLOPs Utilization metrics — how hard the GPU is actually working. |
| `--otlp-traces-endpoint` *(ObservabilityConfig)* | — | Send OpenTelemetry traces to a collector (per-request timing). |
| `--enable-log-requests` / `--no-` *(AsyncEngineArgs)* | `False` | Per-request logging (IDs, params; prompts at DEBUG). Replaces the old `--disable-log-requests`. |

---

## 10. How many users per server — concurrency (slide 14)

> *Measured: RTX 5090 (32 GB), Qwen 3.5-9B BF16, KV pool ~9 GB @ 90% util. Conservative 80% / Recommended 90% / Aggressive 95%.*

| Flag (group) | Default | What it does |
|---|---|---|
| `--max-num-seqs` *(SchedulerConfig)* | engine-computed | Maximum concurrent sequences per iteration — the hard ceiling on the slot counts in slide 14's table. |
| `--max-num-batched-tokens` *(SchedulerConfig)* | engine-computed | Token budget per iteration; with context length this is where the bottleneck "flips" from compute-bound (<8K) to KV-bound (>16K). |
| `--gpu-memory-utilization` *(CacheConfig)* | `0.92` | The 80 / 90 / 95% columns of the concurrency table. |

---

## 11. Model & precision — Qwen BF16 (slides 13–14)

| Flag (group) | Default | What it does |
|---|---|---|
| `--dtype` *(ModelConfig)* | `auto` | Weight/activation precision. The deck runs **`bfloat16`** ("Qwen 3.5-9B BF16, 18 GB"). `auto` picks BF16 for BF16 models. Choices: `auto, bfloat16, float16/half, float32/float`. |
| `--quantization`, `-q` *(ModelConfig)* | from config | Weight quantization method — fits larger models on consumer GPUs (slide 19's "top-tier quality without top-tier hardware"). `--dtype half` is recommended for AWQ. |
| `--load-format` *(LoadConfig)* | `auto` | How weights are loaded; `safetensors` is the modern default, with faster options like `runai_streamer`. |
| `--max-model-len` *(ModelConfig)* | from config | Context window, e.g. `32768` (or `32k`) for the 32K row. |

---

## 12. Time-to-first-word & latency tuning (slide 15)

> *"~120 ms cold / ~40 ms cached at 1K; ~1.9 s / ~600 ms at 16K."*

| Flag (group) | Default | What it does |
|---|---|---|
| `--enforce-eager` / `--no-` *(ModelConfig)* | `False` | Forces eager mode (no CUDA graphs). Leave **off** for best latency (CUDA graphs + eager hybrid); turn on only to save memory or debug. |
| `--performance-mode` *(VllmConfig)* | `balanced` | `interactivity` for lowest first-word latency; `throughput` for max aggregate tokens/sec. |
| `-O` / `--optimization-level` *(CompilationConfig)* | `2` | Startup-time vs runtime-performance trade-off (`-O0` fastest startup … `-O3` best performance). Replaces the old CUDA-graph capture flags such as `--max-seq-len-to-capture`; fine-grained capture lives in `--compilation-config`. |
| `--async-scheduling` / `--no-` *(SchedulerConfig)* | on | Overlaps scheduling with GPU work to remove utilization gaps — better latency and throughput. |
| `--stream-interval` *(SchedulerConfig)* | `1` | Tokens buffered before streaming out; `1` = smoothest streaming, higher = less host overhead. |

---

## 13. Long conversations / 32K context (slide 14)

| Flag (group) | Default | What it does |
|---|---|---|
| `--max-model-len` *(ModelConfig)* | from config | Upper bound on conversation length — the 2K → 32K rows. |
| `--disable-sliding-window` / `--no-` *(ModelConfig)* | `False` | Disable sliding-window attention, capping to the window size — affects how very long contexts are handled. |
| `--hf-overrides` *(ModelConfig)* | `{}` | Pass through HuggingFace config overrides, including **RoPE scaling** to extend context beyond the trained length, e.g. `--hf-overrides '{"rope_scaling":{"rope_type":"dynamic","factor":2.0}}'` (replaces the old top-level `--rope-scaling`). |

---

## 14. "What's next" — speed via speculation (slide 19)

> The closing slide points at faster, smarter small models. Speculative decoding is vLLM's lever here.

| Flag (group) | Default | What it does |
|---|---|---|
| `--speculative-config`, `-sc` *(VllmConfig)* | — | One JSON config for **speculative decoding** (a small draft model proposes tokens a big model verifies → faster generation). Replaces all the old `--speculative-*` / `--ngram-*` flags. Set the method via `--spec-method` (e.g. `ngram`, `eagle`, `mtp`, `draft_model`), the draft via `--spec-model`, and count via `--spec-tokens`. |

```bash
vllm serve <model> \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4}'
```

---

## A production starting point

A single command pulling together the deck's recommendations (RTX 5090, Qwen 9B, ~90% util,
structured output, metrics on). Note prefix caching and chunked prefill are already on by default:

```bash
vllm serve Qwen/Qwen3.5-9B \
  --dtype bfloat16 \
  --max-model-len 32k \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --kv-cache-dtype fp8 \
  --structured-outputs-config.backend xgrammar \
  --kv-cache-metrics \
  --performance-mode interactivity
  # --tensor-parallel-size 4   # add when one GPU isn't enough (slide 12)
```

---

## Arguments not related to the presentation

Documented in the reference but outside the talk's scope:

- **LoRA serving:** `--enable-lora`, `--max-loras`, `--max-lora-rank`, `--lora-extra-vocab-size`,
  `--fully-sharded-loras` *(LoRAConfig)*
- **Multi-modal:** `--limit-mm-per-prompt`, `--mm-processor-kwargs`, `--mm-encoder-tp-mode`,
  `--allowed-local-media-path`, … *(MultiModalConfig)*
- **MoE / expert & data parallel:** `--enable-expert-parallel`, `--data-parallel-size`,
  `--all2all-backend`, `--enable-eplb`, `--expert-placement-strategy` *(ParallelConfig)*
- **Mamba / SSM:** `--mamba-backend`, `--mamba-cache-dtype`, `--mamba-block-size` *(MambaConfig/CacheConfig)*
- **Kernel / weight-transfer / RL internals:** `--moe-backend`, `--linear-backend`,
  `--weight-transfer-config`, `--kv-transfer-config`, `--enable-sleep-mode`
- **Misc:** `--seed`, `--max-logprobs`, `--download-dir`, `--config-format`, `--revision`

---

*Generated from the stable vLLM Engine Arguments reference and `OnPremAI.pptx`.*
