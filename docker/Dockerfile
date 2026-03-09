# Development image for the L3MVN -> iGibson porting workflow.
# Habitat-Sim/Lab, Detectron2, and RedNet are excluded from this image.
# This image provides runtime dependencies only; project code is mounted at runtime.

FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

ENV http_proxy=http://172.18.80.100:8090
ENV https_proxy=http://172.18.80.100:8090
ENV HTTP_PROXY=http://172.18.80.100:8090
ENV HTTPS_PROXY=http://172.18.80.100:8090

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    python3 \
    python3-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libegl1-mesa \
    libegl1-mesa-dev \
    libglew-dev \
    libglfw3-dev \
    libboost-all-dev \
    net-tools \
    iputils-ping \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN python -m pip install --upgrade pip==21.3.1 setuptools==59.6.0 wheel

# Install NumPy first so compiled packages share a compatible C-API.
RUN pip install --no-cache-dir "numpy==1.22.4"

RUN pip install --no-cache-dir \
    https://download.pytorch.org/whl/cu110/torch-1.7.0%2Bcu110-cp38-cp38-linux_x86_64.whl \
    https://download.pytorch.org/whl/cu110/torchvision-0.8.1%2Bcu110-cp38-cp38-linux_x86_64.whl \
    https://download.pytorch.org/whl/torchaudio-0.7.0-cp38-cp38-linux_x86_64.whl

# Core L3MVN dependencies.
RUN pip install --no-cache-dir \
    scikit-fmm==2019.1.30 \
    scikit-learn==0.22.2.post1 \
    scikit-image==0.15.0 \
    ifcfg \
    transformers==4.23.1 \
    tensorboard \
    opencv-python==4.5.5.64 \
    Pillow==8.4.0 \
    scipy==1.7.3 \
    matplotlib==3.5.3

# iGibson dependencies.
RUN pip install --upgrade setuptools==65.5.0 && \
    cd /opt && git clone https://github.com/StanfordVL/iGibson.git && \
    cd iGibson && \
    git submodule update --init --recursive && \
    pip install -e . --no-cache-dir

WORKDIR /workspace

RUN mkdir -p /mount/nas2 /mount/nas3

# GPU and rendering defaults.
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Use EGL for headless rendering.
ENV IGIBSON_RENDERING_BACKEND=egl

ENV PYTHONPATH=/mount/nas2/users/dukim/vla_ws/L3MVN:$PYTHONPATH

RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    pytest

CMD ["/bin/bash"]
