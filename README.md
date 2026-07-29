# GE-GAN: Graph Embedding GAN for Road Traffic State Estimation

[![Paper](https://img.shields.io/badge/DOI-10.1016%2Fj.trc.2020.102635-blue)](https://doi.org/10.1016/j.trc.2020.102635)
[![PeMS Data Downloader](https://img.shields.io/badge/PeMS-data%20downloader-green)](https://github.com/wcc961129/pems-data-downloader)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

Official open-source implementation of:

> Dongwei Xu, Chenchen Wei, Peng Peng, Qi Xuan, and Haifeng Guo.
>
> **GE-GAN: A novel deep learning framework for road traffic state estimation.**
>
> *Transportation Research Part C: Emerging Technologies*, 117, 102635, 2020.

[Read the paper](https://doi.org/10.1016/j.trc.2020.102635)

## Overview

GE-GAN estimates missing road traffic states by combining graph embedding and a generative adversarial network:

1. DeepWalk learns representations of the detector road network.
2. The learned representation selects detectors that are most relevant to a target detector.
3. A Wasserstein GAN learns the traffic-state distribution and estimates the target detector's state.

The paper evaluates the framework on Caltrans PeMS District 7 traffic volume data and a Seattle traffic-speed dataset.

## Repository

```text
GE_GAN/
├── data/                 Published PeMS and Seattle matrices and graph files
├── deepwalk.py           Graph embedding
├── get_data.py           Related-detector selection and data preparation
├── model.py              GE-GAN model
├── main.py               Experiment entry point and parameters
└── visualization.py      Result visualization
```

The implementation is the original research release and uses its historical TensorFlow-era dependency stack. For reproducible work, isolate the environment and record exact dependency versions.

## Download the Caltrans PeMS source data

The standalone [Caltrans PeMS Data Downloader](https://github.com/wcc961129/pems-data-downloader) can prepare the District 7 source period and export matrices compatible with this repository:

```bash
pip install git+https://github.com/wcc961129/pems-data-downloader.git
playwright install chromium

pems-data auth
pems-data fetch \
  --profile ge-gan-d7-2014 \
  --output data/ge-gan-d7-2014
```

The profile covers May 1 through June 30, 2014 and uses the 23-station header published in this repository. It creates:

```text
ge_gan/combine_E_workday_n0.csv
ge_gan/combine_E_weekend_n0.csv
```

This workflow prepares the same PeMS source period and compatible matrix orientation. It does not claim byte-identical historical files or a complete end-to-end reproduction of every paper result. See the downloader's [GE-GAN reproducibility notes](https://github.com/wcc961129/pems-data-downloader/blob/main/docs/ge-gan.md).

## Run the original experiment

From the repository root, review the paths and parameters in `GE_GAN/main.py`, then run:

```bash
python -m GE_GAN.main
```

Important configuration includes:

- `data_flag`: PeMS or Seattle.
- `file_name`: workday or weekend matrix.
- `target_segement`: target detector column.
- `sliding_windows`, `epoch`, `batch_size`: training parameters.
- `select_nums`, `walk_length`, `window_size`: graph embedding parameters.

## Citation

If this repository contributes to your research, please cite:

```bibtex
@article{xu2020gegan,
  title={GE-GAN: A novel deep learning framework for road traffic state estimation},
  author={Xu, Dongwei and Wei, Chenchen and Peng, Peng and Xuan, Qi and Guo, Haifeng},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={117},
  pages={102635},
  year={2020},
  doi={10.1016/j.trc.2020.102635}
}
```

If you use the separate PeMS downloader, cite its software release as well.

## Links

- [Caltrans PeMS Data Downloader](https://github.com/wcc961129/pems-data-downloader)
- [Chenchen Wei's CSDN blog](https://blog.csdn.net/w771792694)
- [Paper on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0968090X19312409)

## License

This project is released under the [Apache License 2.0](LICENSE).
