# torch-m3gnet

**This is an unofficial PyTorch implementation of M3GNet. The authors' reimplementation of MEGNet and M3GNet with PyTorch/DGL will be available [here](https://github.com/materialsvirtuallab/m3gnet-dgl).**

## Limitations

- Only support structures with PBC
- Only support potential prediction
- No support: `state`, `MaterialGraph.phi`

## Installation

See [docs/installation.md](docs/installation.md).

## Preparing benchmark datasets

See [docs/benchmark.md](docs/benchmark.md)

## Development

Docuement
```shell
sphinx-autobuild --host 0.0.0.0 --port 8000 docs docs_build
```

GPU container
```shell
docker run --gpus '"device=1"' -it -v $(pwd):/app -p 6006:6006 m3gnet

# tensorboard
tensorboard --logdir results/ --host 0.0.0.0 --port 6006
```

## References
- M3gNet <https://github.com/materialsvirtuallab/m3gnet>
- NequIP <https://github.com/mir-group/nequip>
- MatDeepLearn <https://github.com/vxfung/MatDeepLearn>
- change_transfer_nnp <https://github.com/pfnet-research/charge_transfer_nnp>
- nn-template <https://github.com/grok-ai/nn-template>
