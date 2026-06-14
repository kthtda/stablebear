# Base is parametrized so the same Dockerfile serves both:
#   - GPU/Linux dev   -> nvidia/cuda:12.6.3-devel-ubuntu22.04 (ships nvcc; the host
#     driver 580 / CUDA 13 is backward-compatible so 12.6 runs fine)
#   - CPU-only / macOS -> ubuntu:22.04 (multi-arch incl. arm64; no nvcc, so the
#     build auto-disables CUDA)
# Both are Ubuntu 22.04 => system python3 is 3.10 (pinned per project request).
# On GPU hosts the RTX 4070 Ti (Ada, sm_89) is detected via CMAKE_CUDA_ARCHITECTURES=native,
# which requires CMake >= 3.24 (pip-installed below, since Jammy's apt cmake is 3.22).
ARG BASE_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Toolchain + Python + Node (for Claude Code) in one layer.
# gcc/g++-13: full C++20 support and the top of CUDA 12.6's supported host-compiler
# range (<= 13.2). Jammy's default gcc-11 has only partial C++20, so we pull 13
# from the ubuntu-toolchain-r/test PPA and make it the default.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg ca-certificates \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc-13 \
        g++-13 \
        cmake \
        ninja-build \
        git \
        git-lfs \
        openssh-client \
        curl \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        sudo \
        less \
        ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
         --slave /usr/bin/g++ g++ /usr/bin/g++-13 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-13 100

# Pre-trust GitHub's host keys so SSH git operations don't hang on an interactive
# host-key prompt. The SSH client itself is openssh-client (installed above); VS
# Code forwards the host SSH agent into the container (SSH_AUTH_SOCK), so keys are
# already available -- only the client + known_hosts were missing.
RUN ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /etc/ssh/ssh_known_hosts 2>/dev/null || true

# GitHub CLI (gh) from the official apt repo (not in Ubuntu's archives). Auth
# comes from the host: devcontainer.json bind-mounts ~/.config/gh and forwards
# GH_TOKEN, so no in-container login is needed when the host is set up.
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 LTS via NodeSource (for @anthropic-ai/claude-code).
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

# Claude Code is installed npm-global as root (npm prefix is /usr), so the
# non-root 'ubuntu' runtime user can't write to /usr/lib/node_modules and its
# background auto-updater fails with "no write permission to npm prefix". The
# image pins the version, so disable the background updater; bump Claude Code by
# rebuilding the image (or switch to the native installer, which installs into a
# user-writable ~/.local and self-updates, if in-container updates are wanted).
ENV DISABLE_AUTOUPDATER=1

# Make CUDA discoverable system-wide, not just in env-inheriting shells. The base
# nvidia/cuda image exports /usr/local/cuda/bin on PATH via docker ENV, which
# interactive container terminals inherit -- but login shells and any process that
# runs with a sanitized environment (some VS Code task runners, build tools) drop
# it and then "nvcc: command not found". A profile.d script + explicit ENV pin the
# toolchain path, CUDA_HOME (for CMake/scikit-build detection), and lib64 so it is
# found regardless of how the process was launched. Guarded with `[ -d ]` so the
# CPU-only base image (no /usr/local/cuda) is a harmless no-op.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
RUN if [ -d /usr/local/cuda ]; then \
        printf '%s\n' \
            'if [ -d /usr/local/cuda ]; then' \
            '  export CUDA_HOME=/usr/local/cuda' \
            '  export PATH="$CUDA_HOME/bin:$PATH"' \
            '  export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' \
            'fi' \
            > /etc/profile.d/cuda.sh \
        && chmod 0644 /etc/profile.d/cuda.sh; \
    fi

# 22.04 has no uid-1000 user (unlike 24.04); create one with passwordless sudo.
RUN useradd --create-home --shell /bin/bash --uid 1000 ubuntu \
    && echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

# Shared venv (isolates from system python; matches Python 3.10).
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV" && chown -R ubuntu:ubuntu "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

USER ubuntu
RUN pip install --upgrade pip

# Modern CMake into the venv (on PATH ahead of /usr/bin, so it shadows Jammy's
# apt cmake 3.22). The minimal-module build sets CMAKE_CUDA_ARCHITECTURES=native
# to target the host GPU (sm_89 on the RTX 4070 Ti); the `native` keyword was
# added in CMake 3.24, so 3.22 emits an empty `compute_,code=sm_` and nvcc dies
# with "Unsupported gpu architecture 'compute_'".
RUN pip install 'cmake>=4,<5'

# Install Python deps straight from the repo's canonical sources, so there's
# no second copy to keep in sync:
#   - runtime + test deps -> pyproject.toml ([project].dependencies and
#     [tool.cibuildwheel].test-requires), extracted with tomli (py3.10 has no
#     stdlib tomllib)
#   - docs (Sphinx) deps  -> docs/requirements.txt
# The cmake build's tooling (numpy, pybind11-stubgen) arrives via these too.
# The C++/CUDA extension itself is NOT pip-installed here; it is built and
# symlinked into stablebear/ by `cmake --install` in postCreateCommand.
COPY --chown=ubuntu pyproject.toml /tmp/build/pyproject.toml
COPY --chown=ubuntu docs/requirements.txt /tmp/build/docs-requirements.txt
RUN pip install tomli \
    && python3 -c "import tomli; d=tomli.load(open('/tmp/build/pyproject.toml','rb')); print('\n'.join(d['project']['dependencies'] + d.get('tool',{}).get('cibuildwheel',{}).get('test-requires',[])))" > /tmp/build/pyproject-deps.txt \
    && pip install -r /tmp/build/pyproject-deps.txt -r /tmp/build/docs-requirements.txt
