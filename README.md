# torch-m3gnet
PyTorch implementation of M3GNet

## Installation

### Local

```shell
conda create -n m3gnet python=3.10 pip
conda activate m3gnet
pip install -e ".[dev,docs]"
```

### GPU container (w/ pytorch)

```shell
```

### GPU container for original M3GNet package (w/ tensorflow)

```shell
singularity build --fakeroot tensorflow.sif containers/m3gnet_tensorflow.def
# sudo singularity build --sandbox tensorflow.sif containers/m3gnet_tensorflow.def
singularity run --nv tensorflow.sif
```

## References
- M3gNet <https://github.com/materialsvirtuallab/m3gnet>
- NequIP <https://github.com/mir-group/nequip>
