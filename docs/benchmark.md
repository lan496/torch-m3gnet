# Benchmark

## IAP datasets

{cite}`doi:10.1021/acs.jpca.9b08723`

```shell
git clone git@github.com:materialsvirtuallab/mlearn.git datasets/

# CPU
python scripts/load_mlearn_dataset.py --raw_datadir datasets/mlearn/data --element Cu --config_path configs/mlearn_Cu.yaml --num_workers 1

# GPU
python scripts/load_mlearn_dataset.py --raw_datadir datasets/mlearn/data --element Cu --config_path configs/mlearn_Cu.yaml --device cuda --num_workers 1
```

## MPF.2021.2.8

<https://figshare.com/articles/dataset/MPF_2021_2_8/19470599>

```shell
git clone git@github.com:materialsvirtuallab/mlearn.git datasets/

# CPU
python scripts/load_mpf_dataset.py --raw_datadir datasets/ --config_path configs/mpf.yaml --num_workers 1

# GPU
python scripts/load_mpf_dataset.py --raw_datadir datasets/ --config_path configs/mpf.yaml --device cuda --num_workers 1
```

## Phonon dispersion curve

<https://figshare.com/articles/dataset/m3gnet_phonon_dispersion_curve_of_328_materials/20217212>

## References

```{bibliography}
:filter: docname in docnames
```
