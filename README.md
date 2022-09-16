# torch-m3gnet
[![testing](https://github.com/lan496/torch-m3gnet/actions/workflows/testing.yml/badge.svg)](https://github.com/lan496/torch-m3gnet/actions/workflows/testing.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/lan496/torch-m3gnet/main.svg)](https://results.pre-commit.ci/latest/github/lan496/torch-m3gnet/main)
[![codecov](https://codecov.io/gh/lan496/torch-m3gnet/branch/main/graph/badge.svg?token=I1J0RYFGP8)](https://codecov.io/gh/lan496/torch-m3gnet)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

🚧This repository is heavily under construction!🚧

**This is an unofficial PyTorch implementation of M3GNet. The authors' reimplementation of MEGNet and M3GNet with PyTorch/DGL will be available [here](https://github.com/materialsvirtuallab/m3gnet-dgl).**

- Documentation <https://lan496.github.io/torch-m3gnet/>

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
