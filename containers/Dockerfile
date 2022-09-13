# https://hub.docker.com/r/nvidia/cuda/tags
FROM nvidia/cuda:10.2-devel-ubuntu18.04

# Configure for tzdata
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install -y --no-install-recommends --fix-missing \
        # Prerequisites for python
        make \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        # Others
        cmake \
        git && \
    rm -rf /var/lib/apt/lists/*

# Create workspace
RUN mkdir /app
WORKDIR /app

# Create user and group
ARG USERNAME=app
ARG GROUPNAME=app
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME
USER $USERNAME

# pyenv
ARG PYTHON_VERSION="3.10.4"
ENV HOME="/home/${USERNAME}"
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc && eval "$(pyenv init -)"
RUN pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION}
RUN pyenv rehash

# Install dependencies
ARG TORCH="1.12.0"
ARG CUDA="cu102"
RUN pip install --upgrade pip
# https://pytorch.org/get-started/locally/
RUN pip install torch==${TORCH} --extra-index-url https://download.pytorch.org/whl/${CUDA}
# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
RUN pip install  torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-geometric

ENTRYPOINT ["/bin/bash"]
