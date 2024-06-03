# TrafficBots V1.5: TrafficBots + HPTR

This repository includes an updated version of [TrafficBots](https://github.com/zhejz/TrafficBots), which replaces the old [SceneTransformer](https://arxiv.org/abs/2106.08417) backbone with the new [HPTR](https://github.com/zhejz/HPTR) architecture and achieves better performance.

In this repository you will find the training, validation, testing and visualization code for the Waymo Open Sim Agents Challenge ([WOSAC](https://waymo.com/open/challenges/2024/sim-agents/)) and the Waymo Open Dataset Challenges ([WOMD](https://waymo.com/open/challenges/2024/motion-prediction/)) for motion prediction. 

## TrafficBots

<p align="center">
     <img src="docs/trafficbots_banner.jpg" alt="TrafficBots for realistic behavior simulation.", width=620px>
     <br/> <B>TrafficBots</B>, a multi-agent policy that generates realistic behaviors for bot agents by learning from real-world data. <br/>
</p>

> **TrafficBots: Towards World Models for Autonomous Driving Simulation and Motion Prediction**            
> [Zhejun Zhang](https://zhejz.github.io/), [Alexander Liniger](https://alexliniger.github.io/), [Dengxin Dai](https://scholar.google.com/citations?user=T51W57YAAAAJ&hl=en), Fisher Yu and [Luc Van Gool](https://vision.ee.ethz.ch/people-details.OTAyMzM=.TGlzdC8zMjcxLC0xOTcxNDY1MTc4.html).<br/>
> 
> [ICRA 2023](https://ieeexplore.ieee.org/document/10161243)<br/>
> [Project Website with Videos](https://zhejz.github.io/trafficbots)<br/>
> [arXiv Paper](https://arxiv.org/abs/2303.04116)

```bibtex
@inproceedings{zhang2023trafficbots,
  title     = {{TrafficBots}: Towards World Models for Autonomous Driving Simulation and Motion Prediction},
  author    = {Zhang, Zhejun and Liniger, Alexander and Dai, Dengxin and Yu, Fisher and Van Gool, Luc},
  booktitle = {International Conference on Robotics and Automation (ICRA)},
  year = {2023}
}
```

## HPTR: Heterogeneous Polyline Transformer with Relative pose encoding

<p align="center">
     <img src="docs/hptr_banner.png" alt="HPTR realizes real-time and on-board motion prediction without sacrificing the performance.", width=650px>
     <br/><B>HPTR</B> realizes real-time and on-board motion prediction without sacrificing the performance. <br/>
</p>

> **Real-Time Motion Prediction via Heterogeneous Polyline Transformer with Relative Pose Encoding**            
> [Zhejun Zhang](https://zhejz.github.io/), [Alexander Liniger](https://alexliniger.github.io/), [Christos Sakaridis](https://people.ee.ethz.ch/~csakarid/), Fisher Yu and [Luc Van Gool](https://vision.ee.ethz.ch/people-details.OTAyMzM=.TGlzdC8zMjcxLC0xOTcxNDY1MTc4.html).<br/>
> 
> [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/71285)<br/>
> [Project Website](https://zhejz.github.io/hptr)<br/>
> [arXiv Paper](https://arxiv.org/abs/2310.12970)

```bibtex
@inproceedings{zhang2023hptr,
  title = {Real-Time Motion Prediction via Heterogeneous Polyline Transformer with Relative Pose Encoding},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  author = {Zhang, Zhejun and Liniger, Alexander and Sakaridis, Christos and Yu, Fisher and Van Gool, Luc},
  year = {2023},
}
```

## Setup Environment
- Create the [conda](https://docs.conda.io/en/latest/miniconda.html) environment by running `conda env create -f environment.yml`.
- Install [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset) API manually because the pip installation of version 1.6.4 is not supported on some linux, e.g. CentOS. Run 
  ```
  conda activate traffic_bots
  wget https://files.pythonhosted.org/packages/14/99/7d36e6fd9ea4d676d1187c1698f6d837d151ea04fc3172c5c6e9dfa2806d/waymo_open_dataset_tf_2_12_0-1.6.4-py3-none-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
  mv waymo_open_dataset_tf_2_12_0-1.6.4-py3-none-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl waymo_open_dataset_tf_2_12_0-1.6.4-py3-none-any.whl
  pip install --no-deps waymo_open_dataset_tf_2_12_0-1.6.4-py3-none-any.whl
  rm waymo_open_dataset_tf_2_12_0-1.6.4-py3-none-any.whl
  ```
- We use [WandB](https://wandb.ai/) for logging. You can register an account for free.
- Be aware
  - We use 4 *NVIDIA RTX 4090* for training and for evaluation. The training takes at least 5 days to converge, whereas the validation and testing takes around 2-3 days.
  - We cannot share pre-trained models according to the [terms](https://waymo.com/open/terms) of the Waymo Open Motion Dataset.

## Prepare Datasets
- Download the [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/). We use v1.2.0
- Run `python scripts/pack_h5_womd.py` or use [bash/pack_h5.sh](bash/pack_h5.sh) to pack the dataset into h5 files to accelerate data loading during the training and evaluation.
- You should pack three datasets: `training`, `validation` and `testing`. Packing the `training` dataset takes around 2 days. For `validation` and `testing` it takes a few hours. 
- Run `python scripts/pickle_val_scenario.py` to pack the validation dataset into pickle files to accelerate data loading during the validation. This step is necessary and cannot be replaced by `pack_h5` because it is required by the validation API of WOSAC.

## Training
Please refer to [bash/train.sh](bash/train.sh) for the training. The default model corresponds to the [leaderboard submission entry](https://waymo.com/open/challenges/sim-agents/results/5ea7a3eb-7337/1716472677635000/). It has 10M parameters and is trained for 6 epochs, i.e. 0.2 * 6 = 1.2 epochs on the complete WOMD training split. The training takes 5 days on 4 RTX 4090.


## Validation and Testing
Please refer to [bash/submission.sh](bash/submission.sh) for the validation and testing.

Download the file from WandB and submit to the [Waymo Motion Prediction Leaderboard](https://waymo.com/open/challenges/2024/motion-prediction/) or the [Waymo Sim Agents Leaderboard](https://waymo.com/open/challenges/2024/sim-agents/).

## License

This software is made available for non-commercial use under a creative commons [license](LICENSE). You can find a summary of the license [here](https://creativecommons.org/licenses/by-nc/4.0/).

## Acknowledgement

This work is funded by Toyota Motor Europe via the research project [TRACE-Zurich](https://trace.ethz.ch) (Toyota Research on Autonomous Cars Europe).