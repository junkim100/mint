# MINT Dockerfile
# CUDA 12.1 + Python 3.10 + PyTorch + MINT

FROM nvcr.io/nvidia/pytorch:24.09-py3

USER root
ENV DEBIAN_FRONTEND=noninteractive

# 1) System packages
RUN apt-get update -qq \
 && apt-get upgrade -y -qq \
 && apt-get install -y -qq --no-install-recommends \
    git wget curl \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# 2) Upgrade pip
RUN pip install --upgrade pip

# 3) Environment variables
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

# 4) Clone and install MINT in editable mode
WORKDIR /app
RUN git clone https://github.com/junkim100/mint.git mint
WORKDIR /app/mint
RUN pip install --no-cache-dir -e .

# 5) Clone Evalchemy
WORKDIR /app
RUN git clone --recursive https://github.com/junkim100/evalchemy.git evalchemy
WORKDIR /app/evalchemy
RUN pip install --no-cache-dir -e .

# 6) Install evaluation tools
RUN pip install --no-cache-dir \
    'lm-eval[vllm] @ git+https://github.com/junkim100/lm-evaluation-harness'

# 7) Install flash-attention (optional)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "Flash-attention installation failed, continuing without it"

# 8) Install τ-bench from PyPI (official package)
RUN pip install --no-cache-dir tau-bench

# 8b) Verify τ-bench installation
RUN python -c "import tau_bench; print('τ-bench installed successfully')"

# 9) Working directory
WORKDIR /app/mint

# 10) Default command
CMD ["sleep", "infinity"]
