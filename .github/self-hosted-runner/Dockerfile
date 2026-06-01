ARG CUDA_VERSION=12.8.0
ARG PYTHON_VERSION=3.13
ARG LLVM_VERSION=21
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04
ARG PYTHON_VERSION
ARG LLVM_VERSION

RUN apt-get update && apt-get install -y \
    cmake build-essential curl git python3 python3-pip python3-venv jq libicu-dev sudo \
    software-properties-common wget lsb-release gnupg valgrind libc6-dbg lcov \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

# Install LLVM toolchain (clang + coverage tools)
# Must run before switching python3 — llvm.sh uses add-apt-repository which needs system apt_pkg
RUN wget -qO- https://apt.llvm.org/llvm.sh | bash -s -- ${LLVM_VERSION} \
    && update-alternatives --install /usr/bin/llvm-cov llvm-cov /usr/bin/llvm-cov-${LLVM_VERSION} 100 \
    && update-alternatives --install /usr/bin/llvm-profdata llvm-profdata /usr/bin/llvm-profdata-${LLVM_VERSION} 100 \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${LLVM_VERSION} 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${LLVM_VERSION} 100 \
    && rm -rf /var/lib/apt/lists/*

# Switch python3 to the desired version (after LLVM install to avoid breaking apt_pkg)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 100

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages -r /tmp/requirements.txt

RUN useradd -m -s /bin/bash runner \
    && echo "runner ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/runner

WORKDIR /home/runner/actions-runner

ARG RUNNER_VERSION=2.334.0
RUN curl -fsSL -o runner.tar.gz \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    && tar xzf runner.tar.gz \
    && rm runner.tar.gz \
    && ./bin/installdependencies.sh

RUN chown -R runner:runner /home/runner

COPY entrypoint.sh /home/runner/actions-runner/entrypoint.sh
RUN chmod +x /home/runner/actions-runner/entrypoint.sh

USER runner

ENTRYPOINT ["./entrypoint.sh"]
