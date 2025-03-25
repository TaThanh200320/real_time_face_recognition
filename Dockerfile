FROM nvcr.io/nvidia/l4t-tensorrt:r8.2.1-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-pip \
    python3.8-dev \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install \
    Cython==0.29.28 \
    numpy==1.24.4 \
    cmake==3.22.3 \
    opencv-python==4.10.0.84 \
    insightface==0.7.3

COPY . /app
RUN pip3 install onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl
