# Use the CUDA 12.2 base image from NVIDIA
# FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
#FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install JupyterLab
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install jupyterlab
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
RUN pip3 install numpy==2.1.1
RUN pip3 install pandas 
RUN pip3 install matplotlib
RUN pip3 install imagehash
RUN pip3 install tqdm
RUN pip3 install wandb
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install opencv-python

ADD . /src/stsbench
EXPOSE 8888

# Add Jupyter Notebook config

# By default start running jupyter notebook
WORKDIR /src
ENTRYPOINT ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--no-browser", "--port=8888", "--NotebookApp.token='1234'", "--notebook-dir='/src'"]
