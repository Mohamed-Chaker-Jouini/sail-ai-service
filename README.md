# Local LLM API — vLLM + FastAPI Gateway

> A self-hosted, GPU-accelerated AI API that runs **Qwen3-14B** locally via [vLLM](https://github.com/vllm-project/vllm), exposed through a lightweight **FastAPI** service. Designed for air-gapped or proxy-only environments with no model downloads at runtime.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Prerequisites](#prerequisites)
5. [Quick Start](#quick-start)
6. [Configuration Reference](#configuration-reference)
7. [AI & Prompting System](#ai--prompting-system)
8. [API Reference](#api-reference)
9. [Docker & Infrastructure](#docker--infrastructure)
10. [Changing the Model](#changing-the-model)
11. [Proxy Setup](#proxy-setup)
12. [GPU Setup (NVIDIA Container Toolkit)](#gpu-setup-nvidia-container-toolkit)
13. [Troubleshooting](#troubleshooting)

---

## Overview

This is a self-hosted AI API gateway you can drop into any project. It:

- Runs **entirely on-premise** — no cloud, no external API calls at inference time
- Serves the [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) reasoning model through [vLLM](https://github.com/vllm-project/vllm)
- Exposes a clean REST API (`/api/ai/chat`) for chat-style completions
- Accepts an arbitrary `context` JSON object alongside user messages, injected verbatim into the system prompt
- Supports corporate **HTTP proxies** at build time without leaking them into the final image

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Host Machine                      │
│                                                          │
│   ┌─────────────────────┐   ┌─────────────────────────┐ │
│   │   llm-api           │   │     vllm-server          │ │
│   │   (FastAPI)         │──▶│   (vLLM OpenAI API)      │ │
│   │   Port: 8000        │   │   Port: 8001             │ │
│   │                     │   │                          │ │
│   │  main.py            │   │  Model: Qwen3-14B        │ │
│   │  llm.py             │   │  /model → host volume    │ │
│   │  prompting.py       │   │  GPU: all NVIDIA         │ │
│   └─────────────────────┘   └─────────────────────────┘ │
│           ▲                          ▲                   │
│           │ HTTP                     │ volume mount      │
│      External                 ~/.cache/huggingface/      │
│      Clients                    hub/Qwen3-14B/           │
└─────────────────────────────────────────────────────────┘
```

**Request flow:**

```
Client → POST /api/ai/chat
          → prompting.py  (inject system prompt + context)
          → llm.py        (call vLLM at /v1/chat/completions)
          → vllm-server   (GPU inference)
          → response back to client
```

---

## Directory Structure

```
your-project/
├── main.py                  # FastAPI app — routes and request models
├── llm.py                   # vLLM client — sends requests to vLLM
├── prompting.py             # System prompt and message builder
├── requirements.txt         # Python dependencies
├── Dockerfile               # API container build
├── docker-compose.yml       # Full stack definition (api + vllm)
├── .env                     # Runtime configuration (ports, model name)
├── run.sh                   # One-shot setup + launch script (start here)
└── README.md
```

---

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Ubuntu 22.04+ | Other Linux distros work; Windows not tested |
| Docker | 24.x+ | With Compose v2 plugin |
| NVIDIA Driver | 535+ | Check with `nvidia-smi` |
| NVIDIA Container Toolkit | Latest | See [GPU Setup](#gpu-setup-nvidia-container-toolkit) |
| VRAM | 28 GB+ | Qwen3-14B in bfloat16 needs ~28 GB; two 16 GB cards work |
| RAM | 32 GB+ | vLLM loads model weights into system RAM before GPU |
| Disk | 30 GB free | Model weights are ~28 GB |
| CPU | Any x86_64 | No minimum; more cores = faster tokenisation |

---

## Quick Start

### 1. Clone the repo

```bash
git clone <your-repo-url> llm-api
cd llm-api
```

### 2. Run everything

```bash
bash run.sh
```

`run.sh` will:
- Validate Docker, NVIDIA runtime, and port availability
- Offer to install the NVIDIA Container Toolkit if not already present
- Offer to download the model if not already on disk
- Pre-pull all required base images through your proxy before Compose touches the network
- Build the API image and start both containers

The API will be live at `http://localhost:8000` once vLLM passes its health check (allow up to 10 minutes for the model to load on first run).

---

## Configuration Reference

All tuneable values live in `.env`. Docker Compose reads this file automatically.

```dotenv
# .env

# ─── Model ────────────────────────────────────────────────
MODEL_NAME=Qwen/Qwen3-14B       # HuggingFace model ID (also the served-model-name)

# ─── Ports ────────────────────────────────────────────────
VLLM_PORT=8001                   # Internal vLLM OpenAI-compatible API port
AI_API_PORT=8000                 # External port for the FastAPI service

# ─── Proxy (build-time only) ──────────────────────────────
# Set these if your build host is behind a corporate proxy.
# They are stripped from the final image automatically.
# HTTP_PROXY=http://10.93.144.53:8080
# HTTPS_PROXY=http://10.93.144.53:8080
```

### vLLM Parameters (docker-compose.yml → `command:`)

These are passed directly to the vLLM server. Edit `docker-compose.yml` to change them.

| Flag | Default | Description |
|---|---|---|
| `--model` | `/model` | Path inside the container to model weights |
| `--served-model-name` | `Qwen/Qwen3-14B` | Model name clients must reference in requests |
| `--port` | `8001` | Internal port vLLM listens on |
| `--max-model-len` | `8192` | Maximum context window in tokens |
| `--reasoning-parser` | `qwen3` | Enables chain-of-thought reasoning extraction |
| `--enable-auto-tool-choice` | _(flag)_ | Lets the model decide when to call tools |
| `--tool-call-parser` | `hermes` | Tool call output format |
| `--gpu-memory-utilization` | `0.90` | Fraction of GPU VRAM given to the model |
| `--dtype` | `bfloat16` | Weight precision — bfloat16 is optimal for Qwen3 |

### FastAPI Parameters (llm.py)

Controlled via environment variables injected by Docker Compose:

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8001` | URL of the vLLM server (use service name inside Docker) |
| `LLM_MODEL` | `Qwen/Qwen3-32B` | Model name sent in every completion request |

> **Note:** `LLM_MODEL` in `docker-compose.yml` overrides the default in `llm.py`. Always set it in `docker-compose.yml` or `.env`.

### Per-Request Parameters (API)

Clients can override these on every call:

| Field | Default | Range | Description |
|---|---|---|---|
| `temperature` | `0.2` | 0.0–2.0 | Sampling temperature. Lower = more deterministic |
| `max_output_tokens` | `600` | 1–8192 | Hard cap on reply length |
| `top_p` | `1.0` | 0.0–1.0 | Nucleus sampling. Lower = fewer token candidates |
| `stream` | `false` | — | Streaming not yet implemented; always returns full response |

---

## AI & Prompting System

### System Prompt (`prompting.py`)

The default system prompt (`SYSTEM_PROMPT` in `prompting.py`) is a placeholder — replace it with whatever role and rules fit your project:

```
You are an AI assistant embedded in <your product>.
<Add your persona, rules, and constraints here.>
```

**To modify the system prompt:** edit `SYSTEM_PROMPT` in `prompting.py`, then rebuild the API container:

```bash
docker compose build api
docker compose up -d api
```

### Context Injection

Every request accepts a `context` dictionary. `prompting.py` serialises it as JSON and appends it to the system message under a `CONTEXT:` header. Pass in whatever structured data your application wants the model to reason over — database records, API responses, document excerpts, user metadata, anything.

Example context payload:

```json
{
  "order_status": {
    "id": "ORD-8821",
    "status": "delayed",
    "updated_at": "2025-06-10T08:14:00Z"
  },
  "user_notes": "Customer flagged urgency"
}
```

### Message Flow

```
build_messages(user_messages, context)
  ├── [0] role: system  → SYSTEM_PROMPT + serialised CONTEXT
  ├── [1] role: user    → first user message (or prior turn history)
  ├── [2] role: assistant → prior assistant turn (if replaying history)
  └── [n] role: user    → latest user message
```

The caller is responsible for maintaining conversation history across turns. Pass all prior messages in the `messages` array.

---

## API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

---

### `GET /api/ai/health`

Returns the service status and active model.

**Response:**

```json
{
  "status": "ok",
  "model": "Qwen/Qwen3-14B",
  "llm_base_url": "http://vllm:8001",
  "time": "2025-06-10T08:00:00.000000+00:00"
}
```

---

### `POST /api/ai/chat`

Send a conversation and receive an AI reply.

**Request body:**

```json
{
  "conversation_id": "optional-string-you-track",
  "messages": [
    { "role": "user", "content": "What changed in the firewall between 08:00 and 09:00?" }
  ],
  "temperature": 0.2,
  "max_output_tokens": 600,
  "top_p": 1.0,
  "stream": false,
  "context": {
    "firewall_snapshot": { ... }
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `conversation_id` | string | No | Opaque ID you supply for tracking; echoed back |
| `messages` | array | Yes | Ordered list of `{role, content}` objects |
| `temperature` | float | No | Defaults to `0.2` |
| `max_output_tokens` | int | No | Defaults to `600` |
| `top_p` | float | No | Defaults to `1.0` |
| `stream` | bool | No | Always `false` (streaming not implemented) |
| `context` | object | No | Arbitrary JSON injected into system context |

**Response:**

```json
{
  "conversation_id": "optional-string-you-track",
  "message": {
    "id": "msg_1749542400000",
    "role": "assistant",
    "content": "Order ORD-8821 was marked delayed at 08:14 UTC. The customer note indicates urgency...",
    "created_at": "2025-06-10T08:00:00.000000+00:00"
  },
  "usage": {
    "prompt_tokens": 312,
    "completion_tokens": 87,
    "total_tokens": 399
  },
  "model": "Qwen/Qwen3-14B"
}
```

**Error codes:**

| Code | Meaning |
|---|---|
| `400` | `messages` array is empty |
| `422` | Request body validation failed (check field ranges) |
| `500` | vLLM returned an error or is unreachable |

---

## Docker & Infrastructure

### Services

| Service | Container | Image | Port |
|---|---|---|---|
| FastAPI | `llm-api` | Built from `Dockerfile` | `8000→8000` |
| vLLM | `vllm-server` | `vllm/vllm-openai:latest` | `8001→8001` |

### Startup Order

The `api` service depends on `vllm` with `condition: service_healthy`. The health check polls `http://localhost:8001/health` every 30 seconds, with a 10-minute `start_period` to allow for model loading. The API will not start until vLLM is ready.

### Volumes

| Host Path | Container Path | Service | Purpose |
|---|---|---|---|
| `~/.cache/huggingface/hub/Qwen3-14B` | `/model` | vllm | Model weights (read-only at runtime) |

### Shared Memory

vLLM requires `ipc: host` and `shm_size: 16gb` for efficient tensor parallelism across GPU workers.

### Useful Commands

```bash
# View live logs for both services
docker compose logs -f

# View only API logs
docker compose logs -f api

# View only vLLM logs (model loading progress)
docker compose logs -f vllm

# Restart only the API (after code changes)
docker compose up -d --build api

# Stop everything
docker compose down

# Stop and remove volumes
docker compose down -v

# Check GPU usage inside the vLLM container
docker exec vllm-server nvidia-smi
```

---

## Changing the Model

To swap to a different model (e.g. `Qwen3-7B` or `Qwen3-32B`):

### 1. Download the new model

```bash
git clone https://huggingface.co/Qwen/Qwen3-7B ~/.cache/huggingface/hub/Qwen3-7B
```

### 2. Update `.env`

```dotenv
MODEL_NAME=Qwen/Qwen3-7B
```

### 3. Update `docker-compose.yml`

- Change the `volumes` path: `~/.cache/huggingface/hub/Qwen3-7B:/model`
- Change `--served-model-name` in the `command` block
- Adjust `--max-model-len` and `--gpu-memory-utilization` as needed
- Update `LLM_MODEL` in the `api` environment block

### 4. Rebuild and restart

```bash
bash run.sh
```

Or, if the stack is already up and only the API changed:

```bash
docker compose up -d --build api
```

### Model Sizing Guide

| Model | VRAM (bfloat16) | Recommended `--gpu-memory-utilization` |
|---|---|---|
| Qwen3-7B | ~14 GB | 0.90 |
| Qwen3-14B | ~28 GB | 0.90 |
| Qwen3-32B | ~64 GB | 0.85–0.90 |

---

## Proxy Setup

The project is designed for environments where the build host sits behind a corporate HTTP proxy.

**Proxy is used at:**
- `docker pull` — `run.sh` pre-pulls `python:3.12-slim` and `vllm/vllm-openai:latest` with the proxy set explicitly in the environment, before Compose runs. This sidesteps Docker's unreliable proxy inheritance during builds.
- `docker build` time — passed as `ARG HTTP_PROXY` / `ARG HTTPS_PROXY` to `pip install`
- NVIDIA toolkit install — used for `apt` and `curl` calls
- Model download — passed as `git -c http.proxy=...`

**Proxy is NOT used at:**
- Container runtime — explicitly stripped from the final image by clearing `HTTP_PROXY`/`HTTPS_PROXY` at the end of the `Dockerfile`
- vLLM inference — `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` prevent any outbound network calls

To change the proxy address, update `.env`:

```dotenv
HTTP_PROXY=http://your-proxy:8080
HTTPS_PROXY=http://your-proxy:8080
```

`run.sh` reads these and threads them through every step that needs them. Also update `docker-compose.yml` → `build.args` to keep the image build in sync.

---

## GPU Setup (NVIDIA Container Toolkit)

`run.sh` handles this automatically — it detects whether the NVIDIA container runtime is registered with Docker and offers to install the toolkit if not.

To trigger it manually on a fresh host, simply run `bash run.sh` and answer **y** when prompted.

**What the install does:**
1. Configures the APT proxy temporarily for the session (if `HTTP_PROXY` is set in `.env`)
2. Fetches the NVIDIA container toolkit GPG key
3. Adds the NVIDIA apt repository
4. Installs `nvidia-container-toolkit`
5. Registers the NVIDIA runtime with Docker (`/etc/docker/daemon.json`)
6. Restarts Docker

After installation, verify the runtime entry is present:

```bash
cat /etc/docker/daemon.json | grep nvidia
```

---

## Troubleshooting

### vLLM never becomes healthy

```bash
docker compose logs vllm
```

Common causes:
- Model path not found → check the volume mount in `docker-compose.yml` matches the actual download path
- Not enough VRAM → reduce `--gpu-memory-utilization` or switch to a smaller model
- NVIDIA runtime not installed → run `bash run.sh` and answer **y** when prompted to install the toolkit

### API returns 500

```bash
docker compose logs api
```

The API logs the full error from vLLM. Usually means vLLM is still loading (`service_healthy` not yet true) or the model name in the request does not match `--served-model-name`.

### `nvidia-smi` not found inside container

The NVIDIA Container Toolkit is not installed or Docker was not restarted after installation. Run:

```bash
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Proxy errors during build

Check that the proxy address in `docker-compose.yml` → `build.args` is reachable from the build host:

```bash
curl -x http://10.93.144.53:8080 https://pypi.org
```

### Out of shared memory

Increase `shm_size` in `docker-compose.yml` (default `16gb`). Required shared memory scales with the number of GPUs and the model size.