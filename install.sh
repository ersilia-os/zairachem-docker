#!/usr/bin/env bash
#
# Interactive installer for ZairaChem.
#
# Walks you through a full setup, as six numbered steps:
#   1. Preflight        — check Docker, compose, conda/mamba
#   2. Python environment — a conda env, or a python -m venv fallback
#   3. Install ZairaChem  — pip install -e .  (plus the optional isaura extra)
#   4. Docker base images — redis, nginx
#   5. Ersilia models     — the default descriptor/projection images (optional)
#   6. Finish             — start the isaura engine (if installed) + summary
#
# Long-running commands show a spinner with the elapsed time and, in grey, the
# underlying command. Their output is hidden on success and shown in full on
# failure. Pass --verbose to stream everything instead.
#
# Safe to re-run: existing environments and images are reused, not rebuilt.
# Run `bash install.sh --help` for non-interactive flags.
#
# Notes:
#   * We drive per-environment commands with `conda run -n <env> …` rather than
#     `conda activate`, because activation is unreliable inside scripts.
#   * On Apple Silicon the ersiliaos/* images are amd64 and run under emulation —
#     that is expected and not a blocker, just slower.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (overridable via flags)
# ---------------------------------------------------------------------------
ENV_NAME="zairachem"
PYTHON_VERSION="3.11"
VENV_PATH=".venv"
ERSILIA_ENV="ersilia"
ASSUME_YES=false
VERBOSE=false
USE_CONDA="auto"      # auto | no
FETCH_MODELS="ask"    # ask | no
MODELS_OVERRIDE=""
ISAURA_FLAG="ask"     # ask | yes | no
DO_BASE_IMAGES=true

# Fallback default model list, used only if it cannot be read from the installed
# package. Keep in sync with ALL_FEATURIZER in zairachem/base/vars.py.
DEFAULT_MODELS="eos3l5f eos8aa5 eos4u6p eos9o72 eos4ex3 eos82v1 eos1klk"

# ---------------------------------------------------------------------------
# Pretty output
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
  YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
  BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi

info()    { printf '%s\n' "  ${BLUE}•${RESET} $*"; }
success() { printf '%s\n' "  ${GREEN}✓${RESET} $*"; }
warn()    { printf '%s\n' "  ${YELLOW}!${RESET} $*" >&2; }
err()     { printf '%s\n' "  ${RED}✗${RESET} $*" >&2; }

SPIN_FRAMES=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
STEP=0
TOTAL=6

# Restore the cursor on any exit (the spinner hides it); only meaningful on a TTY.
trap '[ -t 1 ] && tput cnorm 2>/dev/null || true' EXIT
trap 'err "Installation failed near line $LINENO. See the output above."' ERR

banner() {
  local bar; bar="$(printf '─%.0s' $(seq 1 46))"
  printf '\n%s╭%s╮%s\n' "$BOLD" "$bar" "$RESET"
  printf '%s│%s %-44s %s│%s\n' "$BOLD" "$RESET" "ZairaChem installer" "$BOLD" "$RESET"
  printf '%s│%s %s%-44s%s %s│%s\n' "$BOLD" "$RESET" "$DIM" "environment | package | models" "$RESET" "$BOLD" "$RESET"
  printf '%s╰%s╯%s\n' "$BOLD" "$bar" "$RESET"
}

# section "Title" -> "Step N/6 · Title"
section() {
  STEP=$((STEP + 1))
  printf '\n%sStep %d/%d ·%s %s%s%s\n' "$DIM" "$STEP" "$TOTAL" "$RESET" "$BOLD" "$1" "$RESET"
}

_fmt_elapsed() { printf '%dm%02ds' $(($1 / 60)) $(($1 % 60)); }

# run_step_impl "label" "grey note" <cmd...>  -> runs cmd, returns its exit code.
# TTY: spinner + elapsed + grey note, output captured and shown only on failure.
# --verbose: stream output live. Non-TTY: quiet, dump captured output on failure.
run_step_impl() {
  local label="$1" note="$2"; shift 2
  local logf rc
  logf="$(mktemp "${TMPDIR:-/tmp}/zc-install.XXXXXX")"

  if [ "$VERBOSE" = true ]; then
    printf '  %s %s %s(%s)%s\n' "•" "$label" "$DIM" "$note" "$RESET"
    if "$@"; then rc=0; else rc=$?; fi
    rm -f "$logf"
    if [ "$rc" -eq 0 ]; then printf '  %s %s\n' "${GREEN}✓${RESET}" "$label"
    else printf '  %s %s\n' "${RED}✗${RESET}" "$label"; fi
    return "$rc"
  fi

  if [ ! -t 1 ]; then
    printf '  %s %s %s(%s)%s\n' "•" "$label" "$DIM" "$note" "$RESET"
    if "$@" >"$logf" 2>&1; then rc=0; else rc=$?; fi
    if [ "$rc" -eq 0 ]; then
      printf '  %s %s\n' "${GREEN}✓${RESET}" "$label"
    else
      printf '  %s %s\n' "${RED}✗${RESET}" "$label"; sed 's/^/    | /' "$logf"
    fi
    rm -f "$logf"
    return "$rc"
  fi

  # Interactive TTY: spinner.
  "$@" >"$logf" 2>&1 &
  local pid=$! i=0 start=$SECONDS
  tput civis 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null; do
    printf '\r\033[2K  %s %s  %s(%s) · %s%s' \
      "${SPIN_FRAMES[$i]}" "$label" "$DIM" "$note" "$(_fmt_elapsed $((SECONDS - start)))" "$RESET"
    i=$(((i + 1) % ${#SPIN_FRAMES[@]}))
    sleep 0.1
  done
  if wait "$pid"; then rc=0; else rc=$?; fi
  tput cnorm 2>/dev/null || true
  printf '\r\033[2K'
  if [ "$rc" -eq 0 ]; then
    printf '  %s %s  %s(%s)%s\n' "${GREEN}✓${RESET}" "$label" "$DIM" "$note" "$RESET"
  else
    printf '  %s %s\n' "${RED}✗${RESET}" "$label"; sed 's/^/    | /' "$logf"
  fi
  rm -f "$logf"
  return "$rc"
}

# Hard step: abort the installer if it fails.
run_step() { run_step_impl "$@" || { err "Step failed: $1"; exit 1; }; }
# Soft step: return the exit code so the caller can continue (e.g. model fetch).
run_step_soft() { run_step_impl "$@"; }

usage() {
  cat <<EOF
${BOLD}ZairaChem installer${RESET}

Usage: bash install.sh [options]

Environment:
  --env-name NAME     Conda env name to create/use (default: ${ENV_NAME})
  --python VER        Python version for a new env (default: ${PYTHON_VERSION})
  --no-conda          Skip conda and use a python venv instead
  --venv PATH         venv location when not using conda (default: ${VENV_PATH})

Ersilia models:
  --ersilia-env NAME  Conda env for the ersilia CLI (default: ${ERSILIA_ENV})
  --no-models         Do not fetch any Ersilia model images
  --models "IDS"      Space-separated model ids to fetch (overrides the defaults)

Other:
  --isaura            Install the optional isaura descriptor-cache extra
  --no-isaura         Do not install isaura
  --no-base-images    Do not pull the redis/nginx base images
  --verbose           Stream command output instead of using spinners
  -y, --yes           Accept all defaults (non-interactive)
  -h, --help          Show this help and exit

Examples:
  bash install.sh
  bash install.sh --yes --no-models
  bash install.sh --no-conda --venv .venv --models "eos3l5f eos8aa5"
EOF
}

# ---------------------------------------------------------------------------
# Prompt helpers (honour --yes and non-interactive stdin by taking the default)
# ---------------------------------------------------------------------------
# ask "question" <default y|n>  -> returns 0 for yes, 1 for no
ask() {
  local prompt="$1" default="${2:-y}" reply hint
  if [ "$ASSUME_YES" = true ] || [ ! -t 0 ]; then
    [ "$default" = y ]; return
  fi
  hint="[Y/n]"; [ "$default" = n ] && hint="[y/N]"
  read -r -p "$(printf '  %s %s ' "$prompt" "$hint")" reply || true
  reply="${reply:-$default}"
  case "$reply" in [Yy]*) return 0 ;; *) return 1 ;; esac
}

# read_value "prompt" "default"  -> echoes the chosen value
read_value() {
  local prompt="$1" default="$2" reply
  if [ "$ASSUME_YES" = true ] || [ ! -t 0 ]; then
    printf '%s' "$default"; return 0
  fi
  read -r -p "$(printf '  %s [%s]: ' "$prompt" "$default")" reply || true
  printf '%s' "${reply:-$default}"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --env-name)     ENV_NAME="$2"; shift 2 ;;
      --python)       PYTHON_VERSION="$2"; shift 2 ;;
      --no-conda)     USE_CONDA="no"; shift ;;
      --venv)         VENV_PATH="$2"; shift 2 ;;
      --ersilia-env)  ERSILIA_ENV="$2"; shift 2 ;;
      --no-models)    FETCH_MODELS="no"; shift ;;
      --models)       MODELS_OVERRIDE="$2"; shift 2 ;;
      --isaura)       ISAURA_FLAG="yes"; shift ;;
      --no-isaura)    ISAURA_FLAG="no"; shift ;;
      --no-base-images) DO_BASE_IMAGES=false; shift ;;
      --verbose)      VERBOSE=true; shift ;;
      -y|--yes)       ASSUME_YES=true; shift ;;
      -h|--help)      usage; exit 0 ;;
      *) err "Unknown option: $1"; echo; usage; exit 2 ;;
    esac
  done
}

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
OS=""; ARCH=""; CONDA_BIN=""; HAVE_CONDA=false
DOCKER_PRESENT=false; DOCKER_RUNNING=false; COMPOSE=""

detect_platform() {
  OS="$(uname -s)"; ARCH="$(uname -m)"
  case "$OS" in
    Darwin|Linux) ;;
    *) warn "Unrecognised OS '$OS'. This installer targets macOS and Linux; continuing anyway." ;;
  esac
}

detect_conda() {
  local c
  for c in conda mamba; do
    if command -v "$c" >/dev/null 2>&1; then CONDA_BIN="$c"; HAVE_CONDA=true; return 0; fi
  done
  return 0
}

detect_docker() {
  if command -v docker >/dev/null 2>&1; then
    DOCKER_PRESENT=true
    if docker info >/dev/null 2>&1; then DOCKER_RUNNING=true; fi
  fi
  if docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE="docker-compose"
  fi
}

mark() { [ "$1" = true ] && printf '%s' "${GREEN}✓${RESET}" || printf '%s' "${RED}✗${RESET}"; }

print_preflight() {
  printf '  %s Platform: %s / %s\n' "$(mark true)" "$OS" "$ARCH"
  printf '  %s Docker installed\n' "$(mark $DOCKER_PRESENT)"
  printf '  %s Docker daemon running\n' "$(mark $DOCKER_RUNNING)"
  if [ -n "$COMPOSE" ]; then
    printf '  %s Compose: %s\n' "$(mark true)" "$COMPOSE"
  else
    printf '  %s Compose (docker compose / docker-compose)\n' "$(mark false)"
  fi
  if [ "$HAVE_CONDA" = true ]; then
    printf '  %s Conda: %s\n' "$(mark true)" "$CONDA_BIN"
  else
    printf '  %s conda/mamba (will use python venv)\n' "$(mark false)"
  fi

  if [ "$DOCKER_RUNNING" != true ]; then
    warn "Docker is not running. The environment and package install will still proceed,"
    warn "but base images, model fetching and the isaura engine will be skipped."
    if [ "$OS" = "Darwin" ]; then
      warn "Start Docker Desktop, then re-run this installer to complete those steps."
    else
      warn "Start the Docker Engine (e.g. 'sudo systemctl start docker'), then re-run."
    fi
  fi
}

# ---------------------------------------------------------------------------
# Environment setup -> defines PY() and PIP(), and zenv() for console scripts
# ---------------------------------------------------------------------------
ENV_MODE=""            # conda | venv
PY_PREFIX=(); PIP_PREFIX=()
PY()  { "${PY_PREFIX[@]}" "$@"; }
PIP() { "${PIP_PREFIX[@]}" "$@"; }

# Run a console script (e.g. isaura) inside the zairachem environment.
zenv() {
  if [ "$ENV_MODE" = "conda" ]; then
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" "$@"
  else
    local bin="$1"; shift
    "$VENV_PATH/bin/$bin" "$@"
  fi
}

conda_env_exists() { "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$1"; }

find_system_python() {
  local c
  for c in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1 &&
       "$c" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null; then
      command -v "$c"; return 0
    fi
  done
  return 1
}

setup_environment() {
  if [ "$USE_CONDA" != no ] && [ "$HAVE_CONDA" = true ]; then
    if ask "Create/use a dedicated conda environment for zairachem?" y; then
      ENV_MODE="conda"
    else
      ENV_MODE="venv"
    fi
  else
    if [ "$USE_CONDA" = no ]; then
      info "Using a python venv (--no-conda)."
    else
      warn "conda/mamba not found — falling back to a python venv."
    fi
    ENV_MODE="venv"
  fi

  if [ "$ENV_MODE" = "conda" ]; then
    ENV_NAME="$(read_value "Conda environment name" "$ENV_NAME")"
    if conda_env_exists "$ENV_NAME"; then
      success "Reusing existing conda env '${ENV_NAME}'."
    else
      run_step "Creating conda env '${ENV_NAME}'" \
        "conda create -n ${ENV_NAME} python=${PYTHON_VERSION}" \
        "$CONDA_BIN" create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
    fi
    PY_PREFIX=("$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python)
    PIP_PREFIX=("$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" pip)
  else
    local pybin
    if ! pybin="$(find_system_python)"; then
      err "No suitable system Python (>= 3.10) found for a venv."
      err "Install Python >= 3.10, or install conda/mamba and re-run."
      exit 1
    fi
    info "Using system Python: ${pybin}"
    if [ -d "$VENV_PATH" ]; then
      success "Reusing existing venv at ${VENV_PATH}."
    else
      run_step "Creating venv at ${VENV_PATH}" "python -m venv ${VENV_PATH}" \
        "$pybin" -m venv "$VENV_PATH"
    fi
    PY_PREFIX=("$VENV_PATH/bin/python")
    PIP_PREFIX=("$VENV_PATH/bin/pip")
  fi
}

# ---------------------------------------------------------------------------
# isaura decision (asked before install so it can be a single pip step)
# ---------------------------------------------------------------------------
INSTALL_ISAURA=false
decide_isaura() {
  case "$ISAURA_FLAG" in
    yes) INSTALL_ISAURA=true ;;
    no)  INSTALL_ISAURA=false ;;
    *)   if ask "Install the optional isaura descriptor-cache support?" n; then
           INSTALL_ISAURA=true
         fi ;;
  esac
}

install_zairachem() {
  decide_isaura
  run_step "Upgrading pip" "pip install --upgrade pip" \
    "${PIP_PREFIX[@]}" install --upgrade pip
  if [ "$INSTALL_ISAURA" = true ]; then
    run_step "Installing ZairaChem + isaura" "pip install -e .[isaura]" \
      "${PIP_PREFIX[@]}" install -e ".[isaura]"
  else
    run_step "Installing ZairaChem" "pip install -e ." \
      "${PIP_PREFIX[@]}" install -e .
  fi
}

# ---------------------------------------------------------------------------
# Docker base images
# ---------------------------------------------------------------------------
ensure_image() {
  local img="$1"
  if docker image inspect "$img" >/dev/null 2>&1; then
    printf '  %s %s %s(already present)%s\n' "${GREEN}✓${RESET}" "$img" "$DIM" "$RESET"
  else
    run_step "Pulling ${img}" "docker pull ${img}" docker pull "$img"
  fi
}

pull_base_images() {
  if [ "$DO_BASE_IMAGES" != true ]; then
    info "Skipping base images (--no-base-images)."; return 0
  fi
  if [ "$DOCKER_RUNNING" != true ]; then
    warn "Docker not running — skipping base images (redis, nginx)."; return 0
  fi
  ensure_image "redis:latest"
  ensure_image "nginx:alpine"
}

# ---------------------------------------------------------------------------
# Ersilia CLI + default model images
# ---------------------------------------------------------------------------
ERSILIA_PREFIX=()

ensure_ersilia() {
  # Prefer an ersilia already on PATH; otherwise install it into a dedicated conda env.
  if command -v ersilia >/dev/null 2>&1; then
    ERSILIA_PREFIX=(ersilia)
    info "Using ersilia found on PATH."
    return 0
  fi
  if [ "$HAVE_CONDA" != true ]; then
    warn "ersilia is not installed and conda/mamba is unavailable to install it."
    warn "Install ersilia manually (see https://github.com/ersilia-os/ersilia) and re-run,"
    warn "or fetch images yourself with: docker pull ersiliaos/<id>:latest"
    return 1
  fi
  if ! conda_env_exists "$ERSILIA_ENV"; then
    run_step "Creating conda env '${ERSILIA_ENV}'" \
      "conda create -n ${ERSILIA_ENV} python=${PYTHON_VERSION}" \
      "$CONDA_BIN" create -y -n "$ERSILIA_ENV" "python=${PYTHON_VERSION}"
  else
    success "Reusing existing conda env '${ERSILIA_ENV}'."
  fi
  if ! "$CONDA_BIN" run --no-capture-output -n "$ERSILIA_ENV" ersilia --help >/dev/null 2>&1; then
    run_step "Installing ersilia" "pip install ersilia" \
      "$CONDA_BIN" run --no-capture-output -n "$ERSILIA_ENV" pip install ersilia
  fi
  ERSILIA_PREFIX=("$CONDA_BIN" run --no-capture-output -n "$ERSILIA_ENV" ersilia)
}

resolve_model_list() {
  if [ -n "$MODELS_OVERRIDE" ]; then
    printf '%s' "$MODELS_OVERRIDE"; return 0
  fi
  local models
  models="$(PY -c 'from zairachem.base.vars import ALL_FEATURIZER; print(" ".join(ALL_FEATURIZER))' 2>/dev/null || true)"
  models="$(printf '%s' "$models" | tr -d '\r')"
  [ -z "$models" ] && models="$DEFAULT_MODELS"
  printf '%s' "$models"
}

fetch_models() {
  if [ "$FETCH_MODELS" = "no" ]; then
    info "Skipping model fetch (--no-models)."; return 0
  fi
  if [ "$DOCKER_RUNNING" != true ]; then
    warn "Docker not running — skipping model fetch."; return 0
  fi

  local models; models="$(resolve_model_list)"
  local count; count="$(printf '%s\n' $models | grep -c . || true)"
  info "Default models: ${models}"
  warn "Fetching these images can take a while and use several GB of disk."
  if ! ask "Fetch ${count} Ersilia model image(s) now?" y; then
    info "Skipping model fetch. You can re-run this installer later to fetch them."
    return 0
  fi

  if ! ensure_ersilia; then
    return 0
  fi

  local id img ok=() skipped=() failed=()
  for id in $models; do
    img="ersiliaos/${id}:latest"
    if docker image inspect "$img" >/dev/null 2>&1; then
      printf '  %s %s %s(already present)%s\n' "${GREEN}✓${RESET}" "$id" "$DIM" "$RESET"
      skipped+=("$id")
      continue
    fi
    if run_step_soft "Fetching ${id}" "ersilia fetch ${id} --from_dockerhub" \
         "${ERSILIA_PREFIX[@]}" fetch "$id" --from_dockerhub; then
      ok+=("$id")
    else
      failed+=("$id")
    fi
  done

  printf '\n  %sModel summary%s\n' "$BOLD" "$RESET"
  printf '    fetched:         %s\n' "${ok[*]:-(none)}"
  printf '    already present: %s\n' "${skipped[*]:-(none)}"
  if [ "${#failed[@]}" -gt 0 ]; then
    printf '    %sfailed:          %s%s\n' "$RED" "${failed[*]}" "$RESET"
    warn "Re-run the installer to retry failed models, or fetch manually with:"
    warn "  ersilia fetch <id> --from_dockerhub"
  fi
}

# ---------------------------------------------------------------------------
# Finish: isaura engine + summary
# ---------------------------------------------------------------------------
engine_listening() {
  # Local MinIO S3 API; zairachem considers the engine "running" when this is open.
  (exec 3<>/dev/tcp/127.0.0.1/9000) 2>/dev/null && { exec 3>&- 3<&- 2>/dev/null || true; return 0; }
  return 1
}

start_isaura_engine() {
  [ "$INSTALL_ISAURA" = true ] || return 0
  if [ "$DOCKER_RUNNING" != true ]; then
    warn "Docker not running — not starting the isaura engine."
    info "Start it later with:  ${BOLD}isaura engine --start${RESET}  (in the '${ENV_NAME}' env)"
    return 0
  fi
  run_step "Starting isaura engine" "isaura engine --start" zenv isaura engine --start
  if engine_listening; then
    success "isaura engine listening on 127.0.0.1:9000"
  else
    warn "isaura engine did not become reachable on 127.0.0.1:9000."
    info "Check with:  ${BOLD}isaura engine --status${RESET}"
  fi
}

final_summary() {
  echo
  echo "  Activate the environment and check the install:"
  echo
  if [ "$ENV_MODE" = "conda" ]; then
    printf '    %sconda activate %s%s\n' "$BOLD" "$ENV_NAME" "$RESET"
  else
    printf '    %ssource %s/bin/activate%s\n' "$BOLD" "$VENV_PATH" "$RESET"
  fi
  printf '    %szairachem --help%s\n' "$BOLD" "$RESET"
  echo
  if [ "$DOCKER_RUNNING" != true ]; then
    warn "Docker was not running: start it and re-run this installer to pull images/models."
  fi
  success "Done."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  parse_args "$@"
  cd "$(dirname "${BASH_SOURCE[0]}")"

  banner

  detect_platform
  detect_conda
  detect_docker

  section "Preflight";            print_preflight
  section "Python environment";   setup_environment
  section "Install ZairaChem";    install_zairachem
  section "Docker base images";   pull_base_images
  section "Ersilia models";       fetch_models
  section "Finish";               start_isaura_engine; final_summary
}

main "$@"
