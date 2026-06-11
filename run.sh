#!/usr/bin/env bash
# =============================================================================
#  Local LLM API — Setup & Launch Script
#  Validates the environment, optionally installs dependencies,
#  downloads the model, and starts the full Docker Compose stack.
# =============================================================================
set -euo pipefail

# ─── Colour palette ──────────────────────────────────────────────────────────
RESET="\033[0m"
BOLD="\033[1m"
DIM="\033[2m"

RED="\033[0;31m"
YELLOW="\033[0;33m"
MAGENTA="\033[0;35m"
CYAN="\033[0;36m"

BGREEN="\033[1;32m"
BRED="\033[1;31m"
BYELLOW="\033[1;33m"
BCYAN="\033[1;36m"
BBLUE="\033[1;34m"
BWHITE="\033[1;37m"

# ─── Helpers ─────────────────────────────────────────────────────────────────
print_banner() {
  echo -e ""
  echo -e "${BCYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
  echo -e "${BCYAN}║${BWHITE}        Local LLM API — vLLM + FastAPI Gateway    ${BCYAN}║${RESET}"
  echo -e "${BCYAN}║${DIM}              Self-hosted · GPU-accelerated · Air-gap ready     ${RESET}${BCYAN}║${RESET}"
  echo -e "${BCYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
  echo -e ""
}

log_step() {
  echo -e "\n${BBLUE}▶${RESET} ${BOLD}$1${RESET}"
}

log_info() {
  echo -e "  ${CYAN}•${RESET} $1"
}

log_ok() {
  echo -e "  ${BGREEN}✔${RESET} $1"
}

log_warn() {
  echo -e "  ${BYELLOW}⚠${RESET}  ${YELLOW}$1${RESET}"
}

log_error() {
  echo -e "  ${BRED}✖${RESET} ${RED}$1${RESET}"
}

log_section() {
  echo -e ""
  echo -e "${BOLD}${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo -e "${BOLD}${MAGENTA}  $1${RESET}"
  echo -e "${BOLD}${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
}

ask_yes_no() {
  # ask_yes_no "Question text" → returns 0 for yes, 1 for no
  local prompt="$1"
  while true; do
    echo -ne "  ${BYELLOW}?${RESET}  ${BOLD}${prompt}${RESET} ${DIM}[y/N]${RESET} "
    read -r answer
    case "${answer,,}" in
      y|yes) return 0 ;;
      n|no|"") return 1 ;;
      *) log_warn "Please answer y or n." ;;
    esac
  done
}

die() {
  log_error "$1"
  echo -e ""
  exit 1
}

# ─── Load .env ────────────────────────────────────────────────────────────────
load_env() {
  if [[ -f ".env" ]]; then
    # Export only non-comment, non-empty lines
    set -o allexport
    # shellcheck source=/dev/null
    source <(grep -v '^\s*#' .env | grep -v '^\s*$')
    set +o allexport
    log_ok "Loaded configuration from .env"
  else
    log_warn ".env not found — using built-in defaults"
  fi

  # Apply defaults if not set
  MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
  VLLM_PORT="${VLLM_PORT:-8001}"
  AI_API_PORT="${AI_API_PORT:-8000}"

  # Derive short model directory name (last path component)
  MODEL_DIR_NAME="${MODEL_NAME##*/}"                    # e.g. Qwen3-14B
  MODEL_PATH="${HOME}/.cache/huggingface/hub/${MODEL_DIR_NAME}"
}

# ─── Checks ──────────────────────────────────────────────────────────────────
check_docker() {
  log_step "Checking Docker"
  if ! command -v docker &>/dev/null; then
    die "Docker is not installed. Please install Docker Engine 24+ and re-run."
  fi

  local version
  version=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
  log_ok "Docker found: v${version}"

  if ! docker compose version &>/dev/null; then
    die "Docker Compose v2 plugin not found. Run: sudo apt-get install docker-compose-plugin"
  fi

  local compose_version
  compose_version=$(docker compose version --short 2>/dev/null || echo "unknown")
  log_ok "Docker Compose found: v${compose_version}"

  if ! docker info &>/dev/null; then
    die "Cannot connect to Docker daemon. Is the Docker service running?
       Try: sudo systemctl start docker
       Or add your user to the docker group: sudo usermod -aG docker \$USER"
  fi
}

check_nvidia() {
  log_step "Checking NVIDIA GPU"

  if ! command -v nvidia-smi &>/dev/null; then
    log_warn "nvidia-smi not found — GPU checks skipped (is the NVIDIA driver installed?)"
    return
  fi

  local gpu_info
  gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
  if [[ -z "${gpu_info}" ]]; then
    log_warn "No NVIDIA GPU detected by nvidia-smi."
  else
    while IFS= read -r line; do
      log_ok "GPU: ${line}"
    done <<< "${gpu_info}"
  fi

  # Check NVIDIA container runtime
  if docker info 2>/dev/null | grep -q "nvidia"; then
    log_ok "NVIDIA container runtime is registered with Docker"
  else
    log_warn "NVIDIA container runtime not detected in Docker."
    if ask_yes_no "Install NVIDIA Container Toolkit now?"; then
      install_nvidia_toolkit
    else
      log_warn "Skipping NVIDIA toolkit install. GPU containers may fail."
    fi
  fi
}

check_model() {
  log_step "Checking model: ${MODEL_NAME}"
  log_info "Expected path: ${MODEL_PATH}"

  if [[ -d "${MODEL_PATH}" ]]; then
    local size
    size=$(du -sh "${MODEL_PATH}" 2>/dev/null | cut -f1 || echo "unknown")
    log_ok "Model directory found (${size})"
  else
    log_warn "Model not found at ${MODEL_PATH}"
    if ask_yes_no "Download ${MODEL_NAME} now? (requires internet / proxy access)"; then
      download_model
    else
      log_warn "Skipping model download."
      log_warn "The vLLM container will fail to start without the model weights."
      log_warn "Re-run run.sh when ready, or set MODEL_NAME in .env and re-run."
    fi
  fi
}

check_compose_file() {
  log_step "Checking docker-compose.yml"
  if [[ ! -f "docker-compose.yml" ]]; then
    die "docker-compose.yml not found in the current directory."
  fi
  log_ok "docker-compose.yml found"
}

check_dockerfile() {
  log_step "Checking Dockerfile"
  if [[ ! -f "Dockerfile" ]]; then
    die "Dockerfile not found in the current directory."
  fi
  log_ok "Dockerfile found"
}

check_ports() {
  log_step "Checking port availability"
  local conflict=0

  for port in "${AI_API_PORT}" "${VLLM_PORT}"; do
    if ss -tlnp 2>/dev/null | grep -q ":${port} " || \
       netstat -tlnp 2>/dev/null | grep -q ":${port} "; then
      log_warn "Port ${port} is already in use."
      conflict=1
    else
      log_ok "Port ${port} is free"
    fi
  done

  if [[ "${conflict}" -eq 1 ]]; then
    log_warn "One or more ports are in use. This may cause docker compose to fail."
    log_warn "Change AI_API_PORT / VLLM_PORT in .env to use different ports."
  fi
}

# ─── Install NVIDIA Container Toolkit ────────────────────────────────────────
install_nvidia_toolkit() {
  log_section "Installing NVIDIA Container Toolkit"

  local proxy=""
  if [[ -n "${HTTP_PROXY:-}" ]]; then
    proxy="${HTTP_PROXY}"
    log_info "Using proxy: ${proxy}"
  fi

  # Temporary APT proxy
  if [[ -n "${proxy}" ]]; then
    echo "Acquire::http::Proxy \"${proxy}\";"  | sudo tee /etc/apt/apt.conf.d/95proxies > /dev/null
    echo "Acquire::https::Proxy \"${proxy}\";" | sudo tee -a /etc/apt/apt.conf.d/95proxies > /dev/null
    log_ok "APT proxy configured"
  fi

  # GPG key
  log_info "Fetching NVIDIA GPG key..."
  if [[ -n "${proxy}" ]]; then
    curl -fsSL -x "${proxy}" https://nvidia.github.io/libnvidia-container/gpgkey \
      | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  else
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  fi
  log_ok "GPG key installed"

  # Repository
  log_info "Adding NVIDIA apt repository..."
  if [[ -n "${proxy}" ]]; then
    curl -s -L -x "${proxy}" https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
  else
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
  fi
  log_ok "Repository added"

  # Install
  log_info "Running apt-get update && install..."
  sudo apt-get update -qq
  sudo apt-get install -y nvidia-container-toolkit
  log_ok "nvidia-container-toolkit installed"

  # Configure Docker runtime
  log_info "Configuring Docker runtime..."
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  log_ok "Docker restarted with NVIDIA runtime"

  # Verify the runtime entry was written — no container pull needed
  local runtime_cfg="/etc/docker/daemon.json"
  if [[ -f "${runtime_cfg}" ]] && grep -q "nvidia" "${runtime_cfg}"; then
    log_ok "NVIDIA runtime entry confirmed in ${runtime_cfg}"
  else
    log_warn "Could not confirm NVIDIA entry in ${runtime_cfg} — check manually."
  fi
}

# ─── Download model ───────────────────────────────────────────────────────────
download_model() {
  log_section "Downloading Model: ${MODEL_NAME}"

  local hf_repo="https://huggingface.co/${MODEL_NAME}"
  log_info "Source: ${hf_repo}"
  log_info "Destination: ${MODEL_PATH}"

  mkdir -p "$(dirname "${MODEL_PATH}")"

  local proxy_args=()
  if [[ -n "${HTTP_PROXY:-}" ]]; then
    log_info "Using proxy: ${HTTP_PROXY}"
    proxy_args=(-c "http.proxy=${HTTP_PROXY}")
  fi

  if command -v git &>/dev/null; then
    if [[ ${#proxy_args[@]} -gt 0 ]]; then
      git "${proxy_args[@]}" clone "${hf_repo}" "${MODEL_PATH}"
    else
      git clone "${hf_repo}" "${MODEL_PATH}"
    fi
    log_ok "Model downloaded to ${MODEL_PATH}"
  else
    die "git is required to download the model. Install with: sudo apt-get install git"
  fi
}

# ─── Pre-pull base images through proxy ──────────────────────────────────────
# Docker Compose build does not reliably inherit HTTP_PROXY for image pulls.
# Pulling explicitly beforehand with the proxy set in the environment ensures
# the layers are in the local cache before Compose ever touches the network.
pull_base_images() {
  log_step "Pre-pulling base images through proxy"

  local proxy="${HTTP_PROXY:-}"

  # Images that must be available before `docker compose up --build`
  local -a images=(
    "python:3.12-slim"
    "vllm/vllm-openai:latest"
  )

  local pull_env=()
  if [[ -n "${proxy}" ]]; then
    log_info "Using proxy: ${proxy}"
    pull_env=(env "HTTP_PROXY=${proxy}" "HTTPS_PROXY=${proxy}" "NO_PROXY=localhost,127.0.0.1")
  fi

  for image in "${images[@]}"; do
    log_info "Pulling ${image} ..."
    if "${pull_env[@]}" docker pull "${image}"; then
      log_ok "Pulled ${image}"
    else
      log_warn "Failed to pull ${image} — Compose will attempt it during build."
    fi
  done
}

# ─── Build & Launch ───────────────────────────────────────────────────────────
launch_stack() {
  log_section "Building & Starting the Stack"

  log_info "Model:        ${BOLD}${MODEL_NAME}${RESET}"
  log_info "Model path:   ${BOLD}${MODEL_PATH}${RESET}"
  log_info "API port:     ${BOLD}${AI_API_PORT}${RESET}"
  log_info "vLLM port:    ${BOLD}${VLLM_PORT}${RESET}"
  echo ""

  pull_base_images

  log_step "Running: docker compose up --build -d"
  echo ""

  if docker compose up --build -d; then
    echo ""
    log_ok "${BGREEN}Stack started successfully!${RESET}"
    print_endpoints
  else
    echo ""
    die "docker compose failed. Check the logs above for details."
  fi
}

print_endpoints() {
  echo ""
  echo -e "${BCYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
  echo -e "${BCYAN}║${BWHITE}  Services are starting up...                                 ${BCYAN}║${RESET}"
  echo -e "${BCYAN}╠══════════════════════════════════════════════════════════════╣${RESET}"
  echo -e "${BCYAN}║${RESET}                                                              ${BCYAN}║${RESET}"
  echo -e "${BCYAN}║${RESET}  ${BGREEN}API Health:${RESET}  ${CYAN}http://localhost:${AI_API_PORT}/api/ai/health${RESET}          ${BCYAN}║${RESET}"
  echo -e "${BCYAN}║${RESET}  ${BGREEN}Swagger UI:${RESET}  ${CYAN}http://localhost:${AI_API_PORT}/docs${RESET}                   ${BCYAN}║${RESET}"
  echo -e "${BCYAN}║${RESET}  ${BGREEN}vLLM API:${RESET}    ${CYAN}http://localhost:${VLLM_PORT}/v1/models${RESET}                ${BCYAN}║${RESET}"
  echo -e "${BCYAN}║${RESET}                                                              ${BCYAN}║${RESET}"
  echo -e "${BCYAN}║${DIM}  Note: vLLM may take up to 10 minutes to load the model.      ${RESET}${BCYAN}║${RESET}"
  echo -e "${BCYAN}║${DIM}  Monitor progress: docker compose logs -f vllm               ${RESET}${BCYAN}║${RESET}"
  echo -e "${BCYAN}║${RESET}                                                              ${BCYAN}║${RESET}"
  echo -e "${BCYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
  echo ""

  echo -e "${DIM}Useful commands:${RESET}"
  echo -e "  ${CYAN}docker compose logs -f${RESET}          ${DIM}# stream all logs${RESET}"
  echo -e "  ${CYAN}docker compose logs -f api${RESET}      ${DIM}# stream API logs${RESET}"
  echo -e "  ${CYAN}docker compose logs -f vllm${RESET}     ${DIM}# stream vLLM / model loading${RESET}"
  echo -e "  ${CYAN}docker compose down${RESET}             ${DIM}# stop everything${RESET}"
  echo -e "  ${CYAN}docker exec vllm-server nvidia-smi${RESET}  ${DIM}# GPU usage inside vLLM${RESET}"
  echo ""
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
  print_banner

  # Ensure we're in the project root (where docker-compose.yml lives)
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${SCRIPT_DIR}"

  log_section "Loading Configuration"
  load_env

  log_section "Pre-flight Checks"
  check_compose_file
  check_dockerfile
  check_docker
  check_nvidia
  check_ports
  check_model

  echo ""
  if ask_yes_no "All checks done. Build and start the stack now?"; then
    launch_stack
  else
    echo ""
    log_info "Aborted. Run the following when ready:"
    echo -e "  ${CYAN}docker compose up --build -d${RESET}"
    echo ""
  fi
}

main "$@"