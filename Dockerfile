FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TRELLIS_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd

RUN apt-get update && apt-get install -y --no-install-recommends \
    blender \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/miniconda.sh \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda-12.4

RUN git clone https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS.2 \
    && cd /opt/TRELLIS.2 \
    && git checkout "$TRELLIS_COMMIT" \
    && git submodule update --init --recursive \
    && bash -lc '. /opt/TRELLIS.2/setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm'

WORKDIR /app
COPY requirements.txt .
RUN conda run -n trellis2 python -m pip install --no-cache-dir -r requirements.txt

COPY assetpipe ./assetpipe
COPY blender ./blender
COPY config.yaml README.md ./

ENV PATH=/opt/conda/envs/trellis2/bin:/opt/conda/bin:$PATH
ENV PYTHONPATH=/opt/TRELLIS.2:/app
ENV HF_HOME=/workspace/cache/huggingface

ENTRYPOINT ["python", "-m", "assetpipe"]
