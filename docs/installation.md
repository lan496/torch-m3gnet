# Installation

## Local

### PyTorch 1.12

```shell
conda create -n m3gnet python=3.10 pip
conda activate m3gnet
conda install -c conda-forge compilers
python -m pip install "torch==1.12.0" --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-scatter torch-sparse "torch-geometric==2.2.0" -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install -e ".[dev,docs]"
```

## GPU container (docker, w/ pytorch)

```shell
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t m3gnet -f containers/Dockerfile .

docker run --gpus all -it  -v $(pwd):/app -p 6006:6006 --name m3gnet m3gnet
# Specify GPU
# docker run --gpus '"device=1"' -it  -v $(pwd):/app --name m3gnet

# In container
pip install -e ".[dev]"

# Checking
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch_scatter"
```

## GPU container (singularity, w/ pytorch)

```shell
singularity build --fakeroot pytorch.sif containers/m3gnet_pytorch.def
# sudo singularity build --sandbox pytorch.sif containers/m3gnet_pytorch.def
singularity run --nv pytorch.sif

# In container
pip install -e ".[dev]"
```

## GPU container for original M3GNet package (w/ tensorflow)

```shell
singularity build --fakeroot tensorflow.sif containers/m3gnet_tensorflow.def
# sudo singularity build --sandbox tensorflow.sif containers/m3gnet_tensorflow.def
singularity run --nv tensorflow.sif
```
