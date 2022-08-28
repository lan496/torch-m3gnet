# torch-m3gnet
PyTorch implementation of M3GNet

## Limitations

- Only support structures with PBC
- No support: `state`, `MaterialGraph.phi`

## Installation

### Local

#### PyTorch 1.12

```shell
conda create -n m3gnet python=3.10 pip
conda activate m3gnet
python -m pip install torch==1.12.0 --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install -e ".[dev,docs]"
```

### GPU container (w/ pytorch)

```shell
singularity build --fakeroot pytorch.sif containers/m3gnet_pytorch.def
# sudo singularity build --sandbox pytorch.sif containers/m3gnet_pytorch.def
singularity run --nv pytorch.sif
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
- MatDeepLearn <https://github.com/vxfung/MatDeepLearn>
- change_transfer_nnp <https://github.com/pfnet-research/charge_transfer_nnp>
- nn-template <https://github.com/grok-ai/nn-template>
