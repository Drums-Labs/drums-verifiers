#!/usr/bin/env bash
# setup-prime-rl-gpu.sh: prime-rl (pinned) with its GPU extras, plus the drums SDK and this adapter, in one self-contained
# prefix (uv, Python and the venv all live under PREFIX, so a prefix built on one x86_64 Linux host can be rsynced to the
# same path on another). Runs without a GPU (the install needs none); --check needs one.
#
#   gpu/setup-prime-rl-gpu.sh [--prefix /opt/prime-rl-gpu] [--adapter <drums-verifiers checkout or bundle>]
#                             [--sdk <drums-*.whl or SDK dir>] [--check]
#
# What is installed (all pinned by prime-rl a566e5d8's uv.lock, installed with `uv sync --frozen`):
#   extras gpu        torch 2.13 cu130, vLLM 0.32.0 (also needed in the orchestrator process: it encodes images for VLM
#                     rollouts), flash-linear-attention 0.6.0 @ 413ad2c (Qwen3.5's linear-attention layers), flashinfer
#                     0.7.0.post1 (+ cu130 jit cache), torchtitan, mamba-ssm (built with MAMBA_SKIP_CUDA_BUILD=TRUE: its CUDA
#                     ops are only used by Nemotron-H, so no nvcc is needed)
#   extra flash-attn  flash-attn 2.8.3 cu130/torch2.13 prebuilt wheel (FA2: prime-rl's attn="auto" on SM80 / A100)
#   extra kernels     prime_kernels 0.1.0 (imported by the trainer and orchestrator code; its kernels are built for sm_100a and
#                     report unavailable on A100, which is fine)
# --check imports every one of them and runs a CUDA op, then prints the versions (versions.json under PREFIX).
set -euo pipefail
PREFIX=/opt/prime-rl-gpu; ADAPTER=""; SDK=""; CHECK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX=$2; shift 2;; --adapter) ADAPTER=$2; shift 2;; --sdk) SDK=$2; shift 2;; --check) CHECK=1; shift;;
    -h|--help) sed -n 2,20p "$0"; exit 0;; *) echo "unknown option $1" >&2; exit 2;;
  esac
done
PRIME_RL_COMMIT=a566e5d8e2a903f44d901e114d5d510888b93303
UV_VERSION=0.11.9                     # prime-rl's Dockerfile.cuda pins this uv
say() { echo "$(date -u +%FT%TZ) [setup-gpu] $*"; }
sudo mkdir -p "$PREFIX" && sudo chown "$(id -u):$(id -g)" "$PREFIX"
export UV_INSTALL_DIR=$PREFIX/bin UV_PYTHON_INSTALL_DIR=$PREFIX/python UV_CACHE_DIR=$PREFIX/uv-cache UV_PYTHON_PREFERENCE=only-managed
export UV_LINK_MODE=copy MAMBA_SKIP_CUDA_BUILD=TRUE MAMBA_FORCE_BUILD=FALSE CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE
UV=$PREFIX/bin/uv
t0=$(date +%s)
if [ ! -x "$UV" ] || ! "$UV" --version | grep -q " $UV_VERSION"; then
  say "uv $UV_VERSION -> $PREFIX/bin"
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh > "$PREFIX/uv-install.log" 2>&1
fi
cd "$PREFIX"
if [ ! -d prime-rl/.git ]; then
  say "clone prime-rl $PRIME_RL_COMMIT"
  git clone -q https://github.com/PrimeIntellect-ai/prime-rl.git
fi
git -C prime-rl checkout -q "$PRIME_RL_COMMIT"
git -C prime-rl -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init -q -- deps/verifiers deps/renderers deps/pydantic-config deps/prime-envs
"$UV" python install 3.12 >> "$PREFIX/uv-install.log" 2>&1
say "uv sync --frozen --extra gpu --extra flash-attn --extra kernels (log: $PREFIX/uv-sync-gpu.log)"
(cd prime-rl && "$UV" sync --frozen --python 3.12 --extra gpu --extra flash-attn --extra kernels) > "$PREFIX/uv-sync-gpu.log" 2>&1 \
  || { tail -40 "$PREFIX/uv-sync-gpu.log"; say "FAILED: uv sync"; exit 1; }
PY=$PREFIX/prime-rl/.venv/bin/python
if [ -n "$SDK" ]; then
  say "drums SDK from $SDK"; "$UV" pip install -q -p "$PY" "$SDK"
fi
if [ -n "$ADAPTER" ]; then
  rm -rf "$PREFIX/drums-verifiers"
  if [ -f "$ADAPTER" ]; then git clone -q "$ADAPTER" "$PREFIX/drums-verifiers"; else cp -a "$ADAPTER" "$PREFIX/drums-verifiers"; fi
  say "adapter $(git -C "$PREFIX/drums-verifiers" rev-parse HEAD)"
  "$UV" pip install -q -p "$PY" --no-deps -e "$PREFIX/drums-verifiers"
fi
say "install done in $(( $(date +%s) - t0 )) s; venv $(du -sh "$PREFIX/prime-rl/.venv" | cut -f1)"
"$PY" - "$PREFIX" "$CHECK" <<'PY'
import importlib, json, subprocess, sys
prefix, check = sys.argv[1], sys.argv[2] == "1"
out = {"prime_rl_commit": subprocess.run(["git", "-C", f"{prefix}/prime-rl", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
mods = ["torch", "vllm", "fla", "flash_attn", "prime_kernels", "flashinfer", "transformers", "verifiers", "renderers", "prime_rl",
        "drums", "drums_verifiers", "drums_chat"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        out[m] = getattr(mod, "__version__", "ok")
    except Exception as e:  # noqa: BLE001
        out[m] = f"IMPORT FAILED: {type(e).__name__}: {e}"[:300]
if check:
    import torch
    out["cuda"] = {"available": torch.cuda.is_available(), "count": torch.cuda.device_count(), "torch_cuda": torch.version.cuda}
    if torch.cuda.is_available():
        out["cuda"]["name"] = torch.cuda.get_device_name(0); out["cuda"]["capability"] = list(torch.cuda.get_device_capability(0))
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16); out["cuda"]["matmul_ok"] = bool(torch.isfinite(x @ x).all())
        from flash_attn import flash_attn_func
        q = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
        out["cuda"]["flash_attn_ok"] = bool(torch.isfinite(flash_attn_func(q, q, q, causal=True)).all())
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # Qwen3.5 linear attention
            out["cuda"]["fla_gated_delta_import"] = "ok"
        except Exception as e:  # noqa: BLE001
            out["cuda"]["fla_gated_delta_import"] = f"FAILED: {e}"[:300]
json.dump(out, open(f"{prefix}/versions{'-check' if check else ''}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
sys.exit(1 if any(isinstance(out.get(m), str) and out[m].startswith("IMPORT FAILED") for m in ["torch", "vllm", "fla", "flash_attn", "prime_kernels", "prime_rl"]) else 0)
PY
