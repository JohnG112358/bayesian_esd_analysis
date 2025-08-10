FROM nvcr.io/nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Base tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential wget git git-lfs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && rm /tmp/miniconda.sh

# Accept Anaconda defaults-channel ToS up front so non-interactive builds don't fail
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda install -y \
    python=3.10 \
    pytorch torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia

RUN python -m pip install --no-cache-dir \
    plotly weightwatcher tqdm wandb numpy requests

# Workspace
WORKDIR /workspace
