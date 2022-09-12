# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-08.html#rel_22-08
# Ubuntu 20.04 including Python 3.8
# NVIDIA CUDA 11.6.2
# PyTorch 1.12.0a0+bd13bc6
FROM nvcr.io/nvidia/pytorch:22.04-py3

# Configure for tzdata
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install -y --no-install-recommends \
    make \
    cmake \
    gcc \
    g++ \
    bc \
    git

# Install dependencies
RUN pip install --upgrade pip
RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
RUN pip install torch-geometric

# Create user and group
ARG USERNAME=app
ARG GROUPNAME=app
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME
USER $USERNAME

WORKDIR /home/$USERNAME

ENTRYPOINT ["/bin/bash"]
