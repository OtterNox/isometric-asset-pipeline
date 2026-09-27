FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TRELLIS_COMMIT=442aa1e1afb9014e80681d3bf604e8d728a86ee7

RUN apt-get update && apt-get install -y --no-install-recommends \
    blender \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libjpeg-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/miniconda.sh \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

RUN conda create -y --override-channels -c conda-forge \
      -n trellis python=3.10 pip \
    && conda install -y --override-channels -n trellis \
      -c nvidia/label/cuda-11.8.0 -c conda-forge cuda \
    && conda run -n trellis python -m pip install --no-cache-dir \
      torch==2.4.0 torchvision==0.19.0 \
      --index-url https://download.pytorch.org/whl/cu118 \
    && conda run -n trellis python -m pip install \
      xformers==0.0.27.post2 \
      --index-url https://download.pytorch.org/whl/cu118 \
    && conda run -n trellis python -m pip install kaolin \
      -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

RUN git clone https://github.com/microsoft/TRELLIS.git /opt/TRELLIS \
    && cd /opt/TRELLIS \
    && git checkout "$TRELLIS_COMMIT" \
    && git submodule update --init --recursive \
    && sed -i '73s/.*/PLATFORM=cuda/' setup.sh \
    && bash -c 'source /opt/conda/etc/profile.d/conda.sh; conda activate trellis; export CUDA_HOME="$CONDA_PREFIX"; export PATH="$CUDA_HOME/bin:$PATH"; export PIP_NO_BUILD_ISOLATION=1; python -m pip install --upgrade setuptools wheel packaging ninja; cd /opt/TRELLIS; rm -rf /tmp/extensions; . ./setup.sh --basic --diffoctreerast --spconv --mipgaussian --nvdiffrast' \
    && rm -rf /tmp/extensions \
    && conda clean --all -y

COPY requirements.txt /tmp/assetpipe-requirements.txt
RUN conda run -n trellis python -m pip install --no-cache-dir \
      -r /tmp/assetpipe-requirements.txt \
    && rm /tmp/assetpipe-requirements.txt

COPY container_entrypoint.sh /usr/local/bin/assetpipe-entrypoint.sh

ENV PATH=/opt/conda/envs/trellis/bin:/opt/conda/bin:$PATH
ENV ASSETPIPE_APP_DIR=/root/app
ENV ASSETPIPE_DATA_ROOT=/workspace
ENV ASSETPIPE_INSTALL_ROOT=/opt
ENV ASSETPIPE_REPO_URL=https://github.com/OtterNox/isometric-asset-pipeline.git
ENV ASSETPIPE_GIT_REF=main
ENV PYTHONPATH=/opt/TRELLIS:/root/app
ENV HF_HOME=/workspace/models/huggingface
ENV TORCH_HOME=/workspace/models/torch
ENV XDG_CACHE_HOME=/workspace/cache
ENV ATTN_BACKEND=xformers
ENV SPCONV_ALGO=native

WORKDIR /root
ENTRYPOINT ["bash", "/usr/local/bin/assetpipe-entrypoint.sh"]
