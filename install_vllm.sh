#!/bin/bash
# install_vllm.sh
# Installs Python, vLLM, and tests it manually on the host

set -e

# ---- Configuration ----
HTTP_PROXY="http://10.93.144.53:8080"
HTTPS_PROXY="http://10.93.144.53:8080"
NO_PROXY="localhost,127.0.0.1"
VENV_DIR="/opt/vllm-env"
MODEL="Qwen/Qwen3-32B-Instruct"
PORT=8001
HF_TOKEN=""  # Set your HuggingFace token here if model is gated

# ---- Colors ----
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()    { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ---- Check root ----
if [ "$EUID" -ne 0 ]; then
  error "Please run as root or with sudo"
fi

# ---- Export proxy for all subsequent commands ----
export HTTP_PROXY=$HTTP_PROXY
export HTTPS_PROXY=$HTTPS_PROXY
export NO_PROXY=$NO_PROXY
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTPS_PROXY
export no_proxy=$NO_PROXY

# -------------------------------------------------------
# STEP 1 - System dependencies
# -------------------------------------------------------
log "Step 1: Installing system dependencies..."

apt-get update -qq

apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  python3-dev \
  build-essential \
  curl \
  wget \
  git \
  ca-certificates \
  libssl-dev \
  libffi-dev

log "Python version: $(python3 --version)"

# -------------------------------------------------------
# STEP 2 - NVIDIA / CUDA checks
# -------------------------------------------------------
log "Step 2: Checking NVIDIA GPU..."

if ! command -v nvidia-smi &> /dev/null; then
  error "nvidia-smi not found. Make sure NVIDIA drivers are installed on the host."
fi

nvidia-smi
log "GPU check passed."

# -------------------------------------------------------
# STEP 3 - Create virtual environment
# -------------------------------------------------------
log "Step 3: Creating Python virtual environment at $VENV_DIR..."

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip inside the venv
pip install --upgrade pip --proxy $HTTP_PROXY

log "Virtual environment ready."

# -------------------------------------------------------
# STEP 4 - Install vLLM
# -------------------------------------------------------
log "Step 4: Installing vLLM (this may take several minutes)..."

pip install vllm --proxy $HTTP_PROXY

log "vLLM installed: $(pip show vllm | grep Version)"

# -------------------------------------------------------
# STEP 5 - Install HuggingFace CLI
# -------------------------------------------------------
log "Step 5: Installing HuggingFace Hub CLI..."

pip install huggingface_hub --proxy $HTTP_PROXY

# -------------------------------------------------------
# STEP 6 - Download the model
# -------------------------------------------------------
log "Step 6: Downloading model $MODEL from HuggingFace..."

if [ -n "$HF_TOKEN" ]; then
  export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
  hf login --token "$HF_TOKEN" --add-to-git-credential
fi

hf download "$MODEL"

log "Model downloaded."

# -------------------------------------------------------
# STEP 7 - Test vLLM server manually
# -------------------------------------------------------
log "Step 7: Starting vLLM server on port $PORT for testing..."
log "This will run in the foreground. Press Ctrl+C to stop."
echo ""
warn "Watch for the line: INFO: Uvicorn running on http://0.0.0.0:$PORT"
warn "Then open a second terminal and run:"
echo ""
echo "    curl http://localhost:$PORT/v1/models"
echo ""

# Remove proxy for running the server (not needed at runtime)
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT"
