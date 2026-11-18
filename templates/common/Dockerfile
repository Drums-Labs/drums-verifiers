# The trainer image for templates/modal and templates/gpu-box: prime-rl at a pinned commit with its GPU extras, plus
# drums-verifiers and drums-sdk. The same file D1 trained with (prime-rl 167d4c3 on 8 x H200 / B200), with the pins as
# build arguments. Modal builds it with `Image.from_dockerfile` (templates/modal/app.py); a GPU box with
# `docker build -f templates/common/Dockerfile templates/common` (templates/gpu-box/run.sh).
#
# Modal's Dockerfile builder is not Docker: single stage, no BuildKit mounts, no USER, LABEL, VOLUME or EXPOSE.
#
# Base: CUDA 13.0.1 devel on Ubuntu 24.04, the CUDA line prime-rl's own Dockerfile.cuda uses; its lock pulls torch 2.14.0
# (cu130), vLLM 0.33.0, flash-attn 2.8.3, flashinfer and nixl-cu13. CUDA 13.0 covers Blackwell (B200) and Hopper (H200).
# The devel image is used because TileLang and FlashInfer JIT-compile kernels with nvcc at run time.
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

# The pins (rl-config.toml [image] passes the first three). The three editable versions belong to PRIME_RL_COMMIT: they
# are what prime-rl's scripts/docker-editable-pretend-versions.sh prints at that commit. Recompute them when you change it.
ARG PRIME_RL_COMMIT=167d4c3bf1cf9d4444cf1b508f3bc2f231f272dc
# DRUMS_VERIFIERS: a 40-character commit sha or a release (rl-config.toml [image] says which); the placeholder fails the build.
ARG DRUMS_VERIFIERS="drums-verifiers @ git+https://github.com/Drums-Labs/drums-verifiers@PIN_A_COMMIT"
ARG DRUMS_SDK="drums-sdk==0.3.0"
ARG VERIFIERS_VERSION=0.4.2.dev47
ARG RENDERERS_VERSION=0.2.1.dev9
ARG PYDANTIC_CONFIG_VERSION=0.4.3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LC_ALL=C.UTF-8
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin

# Python 3.12 (Ubuntu 24.04's default) with pip: Modal's runtime needs `python` and `pip` on PATH to inject its client.
# The prime-rl venv is kept OFF the PATH on purpose so Modal's client never lands in it (see README, "image").
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git git-lfs build-essential ninja-build pkg-config \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        libnuma-dev libnuma1 libibverbs1 librdmacm1 libhwloc15 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# uv 0.11.9: the version prime-rl's Dockerfile.cuda pins (pyproject requires >= 0.11.1 for the "7 days" exclude-newer form).
RUN curl -LsSf https://astral.sh/uv/0.11.9/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh \
    && uv --version

ENV UV_PYTHON_PREFERENCE=only-system
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# prime-rl at the pinned commit. The submodule URLs in .gitmodules are SSH (git@github.com:...); they are rewritten to
# HTTPS for an anonymous clone. prime-envs is needed only because it is a uv workspace member glob referenced by uv.lock.
#
# verifiers, renderers and pydantic-config are editable workspace members whose version hatch-vcs derives from git tags
# (`git describe`); the prime-envs members require `verifiers>=0.4.2.dev47`. A shallow, tagless clone makes verifiers
# read as 0.0.1.dev1 and `uv sync` fails with "aime24 and wordle are incompatible" (D1's second build attempt).
# So the three editables are cloned with full history and tags, and their `fallback-version` is pinned to the exact
# values prime-rl's own scripts/docker-editable-pretend-versions.sh computes for these commits (at 167d4c3: verifiers
# 0.4.2.dev47, renderers 0.2.1.dev9, pydantic-config 0.4.3), the same way its scripts/docker_application.py does.
RUN git config --global url."https://github.com/".insteadOf "git@github.com:" \
    && git clone https://github.com/PrimeIntellect-ai/prime-rl.git /app \
    && cd /app && git checkout -q ${PRIME_RL_COMMIT} \
    && git submodule update --init -- deps/verifiers deps/renderers deps/pydantic-config \
    && git submodule update --init --depth 1 -- deps/prime-kernels deps/prime-envs \
    && git -C deps/verifiers fetch -q --tags && git -C deps/renderers fetch -q --tags && git -C deps/pydantic-config fetch -q --tags \
    && sed -i "s/^fallback-version = \"[^\"]*\"\$/fallback-version = \"${VERIFIERS_VERSION}\"/" deps/verifiers/pyproject.toml \
    && sed -i "s/^fallback-version = \"[^\"]*\"\$/fallback-version = \"${RENDERERS_VERSION}\"/" deps/renderers/pyproject.toml \
    && sed -i "s/^fallback-version = \"[^\"]*\"\$/fallback-version = \"${PYDANTIC_CONFIG_VERSION}\"/" deps/pydantic-config/pyproject.toml \
    && grep -H "^fallback-version" deps/verifiers/pyproject.toml deps/renderers/pyproject.toml deps/pydantic-config/pyproject.toml \
    && git submodule status \
    && echo "verifiers: $(git -C deps/verifiers describe --tags --long --match 'v[0-9]*.[0-9]*.[0-9]*' --exclude '*dev*')" \
    && echo "renderers: $(git -C deps/renderers describe --tags --long --match 'renderers-v*')"

WORKDIR /app

# Exactly the extras prime-rl's Dockerfile.cuda installs (gpu, dashboard, flash-attn, flash-attn-3, flash-attn-cute,
# disagg, quack, kernels), from the locked resolution. `disagg` is kept because the inference router prime-rl starts by
# default (`vllm-router`) ships in that extra. # flash-attn is a prebuilt cu130/torch2.14 wheel (FLASH_ATTENTION_SKIP_CUDA_BUILD is set in pyproject); mamba-ssm builds
# a tiny shim without CUDA kernels (its kernels are only used by Nemotron-H).
ENV MAMBA_SKIP_CUDA_BUILD=TRUE
ENV MAMBA_FORCE_BUILD=FALSE
ENV CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE
# `--frozen`: install exactly what prime-rl's uv.lock records without re-checking it against the project (the pinned
# fallback-version lines above make `--locked` call the lock stale). flashinfer.ai's index answered 503 for a while on
# 2026-11-16; uv's own retries cover short outages.
ENV UV_HTTP_RETRIES=10
ENV UV_HTTP_TIMEOUT=180
RUN uv sync --frozen --no-dev \
        --extra gpu --extra dashboard --extra flash-attn --extra flash-attn-3 --extra flash-attn-cute \
        --extra disagg --extra quack --extra kernels \
    && rm -rf /root/.cache/uv

# drums-verifiers (DrumsTaskset / DrumsEnv / drums-chat harness) and drums-sdk (import name `drums`).
# Installed from OUTSIDE /app: prime-rl's pyproject sets `exclude-newer = "7 days"`, which uv also applies to
# `uv pip install` run inside the project and would reject any release younger than a week (D1 hit this).
# --no-deps: their dependencies (verifiers, pillow) are already in the venv from the lock, and the editable verifiers
# must not be replaced by a PyPI release.
RUN cd / && uv pip install --python /app/.venv/bin/python --no-deps "${DRUMS_VERIFIERS}" "${DRUMS_SDK}" \
    && /app/.venv/bin/python -c "import importlib.metadata as m; print('drums-verifiers', m.version('drums-verifiers'))" \
    && /app/.venv/bin/python -c "import drums, drums_verifiers, drums_chat, drums_tasks; \
from verifiers.v1.utils.loaders import taskset_class, environment_class, harness_class; \
print(taskset_class('drums-verifiers'), environment_class('drums-verifiers'), harness_class('drums-chat'))"

# Import checks that need no GPU. Hard: the trainer entrypoint, the Qwen3.5 modeling (pulls in fla) and transformers.
# Soft (printed, never fails the build): vllm and flash_attn, whose import on a GPU-less build container may complain
# about the missing driver; check_image / smoke import them on a real machine.
RUN /app/.venv/bin/python -c "import torch, transformers, prime_rl, fla; \
import prime_rl.trainer.models.qwen3_5.modeling_qwen3_5; \
print('torch', torch.__version__, 'transformers', transformers.__version__, 'fla', getattr(fla, '__version__', '?'))" \
    && /app/.venv/bin/rl --help > /dev/null \
    && (/app/.venv/bin/python -c "import vllm, flash_attn; print('vllm', vllm.__version__, 'flash_attn', flash_attn.__version__)" \
        || echo "soft check: vllm/flash_attn import failed on the build container (no GPU); re-checked by check_image")

# The HF cache lives on the weights volume (mounted at /weights); set here so every process agrees.
ENV HF_HOME=/weights/hf
ENV HF_HUB_ETAG_TIMEOUT=500
ENV HF_HUB_DOWNLOAD_TIMEOUT=300
