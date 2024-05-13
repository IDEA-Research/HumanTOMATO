# OpenTMA: Open Text-Motion Alignment Project

🕺 Reproduced by [Ling-Hao Chen](https://lhchen.top/) and [Shunlin Lu](https://shunlinlu.github.io/) (credit also with [TMR](https://mathis.petrovich.fr/tmr/), [SwanHub](https://swanhub.co)). 

**❗️[Highlight]: We provide a [demo](https://swanhub.co/demo/Evan/OpenTMR) for the OpenTMA in [HumanTOMATO](https://github.com/IDEA-Research/HumanTOMATO).** The demo is supported by the [SwanHub](https://swanhub.co) engineering team. Hav a try! 

<p align="center">
  <img src="assets/logo.png"/>
</p>

## ✨ Quick Introduction


OpenTMA is a project that aims to provide a simple and efficient way to align text and motion data. It is designed to be easy to use and flexible, allowing users to align text and motion data in the latent space. 

In the [HumanTOMATO](https://lhchen.top/HumanTOMATO/) (ICML 2024) project, we clarify the importance of how to use the text and motion data to generate motions for the first time. We highlight the two method.

> + Replace your CLIP text encoder with OpenTMA text encoder.
> + Introduce the text-motion alignment supervision to your motion generation model during training. 

## 📢 News

+ **[2024/05/12] We release the OpenTMA training and checkpoints.**

## ☑️ Todo List

- [x] Release the OpenTMA training.
- [x] Release the OpenTMA checkpoints.
- [ ] Support PyPI (`pip install opentma`).

## 🚀 Quick start

### Installation

```bash
pip install -r requirements.txt
```


### Downloading Pretrained Checkpoints

We provide some pretrained checkpoints of OpenTMA for evaluation. Here are two methods to download the checkpoints. 1) You can download the checkpoints from the [Google Drive](https://drive.google.com/drive/folders/1QpwuabWIFpXRMMO4ioqRez3oyIhmHkQ5?usp=share_link). 2) You can download the checkpoints from the [Baidu Drive](https://pan.baidu.com/s/1N9P_2q5d2wUEmrVoPOPbZg?pwd=evan) (pwd: `evan`).



### Usage

```python
# Load text and motion data
import torch
from transformers import AutoTokenizer, AutoModel
from tma.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from tma.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder
from collections import OrderedDict

modelpath = 'distilbert-base-uncased'

textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4)
motionencoder = ActorAgnosticEncoder(nfeats=126, vae = True, num_layers=4)

"""
load model here
You need to normalize the motion data with mean and std.
For motionx, they are stored in './deps/t2m/motionx/vector_623/Comp_v6_KLD01/meta/*.npy'
"""

motion = torch.randn(1, 64, 126)    # B = 1, T = , D = , need normalization
lengths = [64]
print(textencoder(["a man is running"]).loc)
print(motionencoder(motion, lengths).loc)
```


## 🏃 Model Training

### 1. Data Preparation

Our OpenTMA project supports three datasets: [HumanML3D](https://github.com/EricGuo5513/HumanML3D?tab=readme-ov-file#how-to-obtain-the-data), [Motion-X](https://motionx.deepdataspace.com/), and [UniMoCap](https://github.com/LinghaoChan/UniMoCap). 

<details>
  <summary><b> HumanML3D Data Preparation </b></summary>

Please following the instructions in the [HumanML3D](https://github.com/EricGuo5513/HumanML3D?tab=readme-ov-file#how-to-obtain-the-data) repository to download and preprocess the data. The data should be stored in the `./datasets/humanml3d` folder. The path tree should look like this:

```
./OpenTMR/datasets/humanml3d/
├── all.txt
├── Mean.npy
├── new_joints/
├── new_joint_vecs/
├── Std.npy
├── test.txt
├── texts/
├── train.txt
├── train_val.txt
└── val.txt
```

</details>


<details>
  <summary><b> Motion-X Data Preparation </b></summary>

Please following the instructions in the [Motion-X](https://github.com/IDEA-Research/Motion-X?tab=readme-ov-file#-dataset-download) project. And then please follow the [HumanTOMATO](https://github.com/IDEA-Research/HumanTOMATO/tree/main/src/tomato_represenation) repository to preprocess the data into `tomatao` format. The data should be stored in the `./datasets/Motion-X` folder. The path tree should look like this:

```
./OpenTMR/datasets/Motion-X
├── mean_std
│   └── vector_623
│       ├── mean.npy
│       └── std.npy
├── motion_data
│   └── vector_623
│       ├── aist/       (subset_*/*.npy)
│       ├── animation/
│       ├── dance/
│       ├── EgoBody/
│       ├── fitness/
│       ├── game_motion/
│       ├── GRAB/
│       ├── HAA500/
│       ├── humanml/
│       ├── humman/
│       ├── idea400/
│       ├── kungfu/
│       ├── music/
│       └── perform/
├── split
│   ├── all.txt
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
└── texts
    ├── semantic_texts
    │   ├── aist/       (subset_*/*.txt)
    │   ├── animation/
    │   ├── dance/
    │   ├── EgoBody/
    │   ├── fitness/
    │   ├── game_motion/
    │   ├── GRAB/
    │   ├── HAA500/
    │   ├── humanml/
    │   ├── humman/
    │   ├── idea400/
    │   ├── kungfu/
    │   ├── music/
    └───└── perform/
```

</details>


<details>
  <summary><b> UniMoCap Data Preparation </b></summary>

Please following the instructions in the [UniMoCap](https://github.com/LinghaoChan/UniMoCap) repository to download and preprocess the data (HumanML3D, BABEL, and KIT-ML). The data should be stored in the `./datasets/UniMocap` folder. The path tree should look like this:

```
./OpenTMR/datasets/UniMocap
├── all.txt
├── Mean.npy
├── new_joints/     (*.npy)
├── new_joint_vecs/ (*.npy)
├── Std.npy
├── test.txt
├── texts/          (*.txt)
├── train.txt
├── train_val.txt
└── val.txt
```

</details>



### 2. Pretrained Checkpoints Used in the Evaluation 

Here, we provide some pre-traind checkpoints for the evaluation. Here are two methods to download the checkpoints:


<details>
<summary><b> Google Drive</b></summary>


Download the checkpoints from the [Google Drive](https://drive.google.com/drive/folders/1aWpJH4KTXsWnxG5MciLHXPXGBS7vWXf7?usp=share_link) and put them in the `./deps` folder. Please unzip the checkpoints via the following command:
```
unzip *.zip
```
Finally, the path tree should look like this:

```
./deps
├── distilbert-base-uncased/
├── glove/
├── t2m/
└── transforms/
```

</details>


<details>
<summary><b> Baidu Drive</b></summary>


Download the checkpoints from the [Baidu Drive](https://pan.baidu.com/s/1SIwGDX2aDWTR4hLhUHrPlw?pwd=evan ) (pwd: `evan`) and put them in the `./deps` folder. Please unzip the checkpoints via the following command:
```
tar –xvf deps.tar
```
Finally, the path tree should look like this:

```
./deps
├── distilbert-base-uncased/
├── glove/
├── t2m/
└── transforms/
```

</details>



### 3. Training

+ Training on HumanML3D:

```bash
python -m train --cfg configs/configs_temos/H3D-TMR.yaml --cfg_assets configs/assets.yaml --nodebug
```

+ Training on Motion-X:

```bash
python -m train --cfg configs/configs_temos/MotionX-TMR.yaml --cfg_assets configs/assets.yaml --nodebug
```

+ Training on UniMoCap:

```bash
python -m train --cfg configs/configs_temos/UniMoCap-TMR.yaml --cfg_assets configs/assets.yaml --nodebug
```

The checkpoints will be saved in the `./experiments/`. If you would like to the debug mode, please remove the `--nodebug` flag. The best checkpoints often appear in the 100-500th epoch.


## 🧪 Test for Evaluation

Before running the code below, please revise the `retreival.sh` (like `path1` variable) file to set the correct path for the data. This command should be used after training. It will evaluate the performance of the model on the test set with **text and motion embeddings**.

```bash
bash retreival.sh
```
The result will be in a markdown table format.

# 🤝🏼 Citation

If you use this repository for research, you need to cite:
```bash
@article{humantomato,
  title={HumanTOMATO: Text-aligned Whole-body Motion Generation},
  author={Lu, Shunlin and Chen, Ling-Hao and Zeng, Ailing and Lin, Jing and Zhang, Ruimao and Zhang, Lei and Shum, Heung-Yeung},
  journal={arxiv:2310.12978},
  year={2023}
}
```

```bash
@article{chen2023unimocap,
  title={UniMocap: Unifier for BABEL, HumanML3D, and KIT},
  author={Chen, Ling-Hao and UniMocap, Contributors},
  journal={https://github.com/LinghaoChan/UniMoCap},
  year={2023}
}
```

```bash
@inproceedings{petrovich23tmr,
    title     = {{TMR}: Text-to-Motion Retrieval Using Contrastive {3D} Human Motion Synthesis},
    author    = {Petrovich, Mathis and Black, Michael J. and Varol, G{\"u}l},
    booktitle = {International Conference on Computer Vision ({ICCV})},
    year      = {2023}
}
```

```bash
@InProceedings{Guo_2022_CVPR,
    author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
    title     = {Generating Diverse and Natural 3D Human Motions From Text},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {5152-5161}
}
```

```bash
@conference{AMASS2019,
  title = {AMASS: Archive of Motion Capture as Surface Shapes},
  author = {Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and Pons-Moll, Gerard and Black, Michael J.},
  booktitle = {International Conference on Computer Vision},
  pages = {5442--5451},
  month = oct,
  year = {2019},
  month_numeric = {10}
}
```

If you have any question, please contact [Ling-Hao Chen](https://lhchen.top/) (thu [DOT] lhchen [AT] gmail [DOT] com) and [Shunlin Lu](https://shunlinlu.github.io/) (shunilnlu0803 [AT] gmail [DOT] com).
