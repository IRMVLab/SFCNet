<h1 align="center"> Spherical Frustum Sparse Convolution Network for LiDAR Point Cloud Semantic Segmentation </h1>   
  <h3 align="center">NeurIPS 2024</h3>

  <p align="center">
    <strong>Yu Zheng*</strong>
    ·
    <strong>Guangming Wang*</strong>
    ·
    <strong>Jiuming Liu</strong>
      ·
    <strong>Marc Pollefeys</strong>
      ·
    <strong>Hesheng Wang#</strong></p>
 <h5 align="left">TL;DR: We propose the spherical frustum structure to avoid quantized information loss in conventional 2D spherical projection for LiDAR point cloud semantic segmentation.</h5>

[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2311.17491)

### :newspaper:News

- **[26/Sept/2024]** Our Paper has been accepted as a Poster in **NeurIPS 2024**.

<!-- <details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a>Abstract</a>
    </li>
          <li>
      <a>Reference</a>
    </li>
  </ol>
</details> -->

### :page_facing_up: Abstract

LiDAR point cloud semantic segmentation enables the robots to obtain fine-grained semantic information of the surrounding environment. Recently, many works project the point cloud onto the 2D image and adopt the 2D Convolutional Neural Networks (CNNs) or vision transformer for LiDAR point cloud semantic segmentation. However, since more than one point can be projected onto the same 2D position but only one point can be preserved, the previous 2D projection-based segmentation methods suffer from inevitable quantized information loss, which results in incomplete geometric structure, especially for small objects. To avoid quantized information loss, in this paper, we propose a novel spherical frustum structure, which preserves all points projected onto the same 2D position. Additionally, a hash-based representation is proposed for memory-efficient spherical frustum storage. Based on the spherical frustum structure, the Spherical Frustum sparse Convolution (SFC) and Frustum Farthest Point Sampling (F2PS) are proposed to convolve and sample the points stored in spherical frustums respectively. Finally, we present the Spherical Frustum sparse Convolution Network (SFCNet) to adopt 2D CNNs for LiDAR point cloud semantic segmentation without quantized information loss. Extensive experiments on the SemanticKITTI and nuScenes datasets demonstrate that our SFCNet outperforms previous 2D projection-based semantic segmentation methods based on conventional spherical projection and shows better performance on small object segmentation by preserving complete geometric structure. 

### :page_with_curl: Results & Pretrained SFCNet models

| dataset       | Val mIoU |                 Download                 |
| ------------- | :------: | :--------------------------------------: |
| SemanticKITTI |   62.9   |  [Model Weight](https://pan.sjtu.edu.cn/web/share/47c40a8d0d69270148011339a5e39fd6)   |
| nuScenes      |   75.9   | [Model Weight](https://pan.sjtu.edu.cn/web/share/07479bdff07258c01099daf8724942aa) |

### :car: Dataset Preparation

#### SemanticKITTI

Download the SemanticKITTI dataset from [official](https://semantic-kitti.org/) and change the dataset path [here](SFCNet\pp_dataset\semkitti_trainset_spp.py).

#### nuScenes

Install the [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit) with

```bash
pip install nuscenes-devkit
```

Use `SFCNet/pp_dataset/generate_nuscenes_datas.py` to generate the file list of nuScenes dataset. First, modify the nuScenes dataset 

```bash
cd SFCNet/pp_dataset/
python generate_nuscenes_datas.py
```

The generated file list will be saved in `SFCNet/pp_dataset/nuscenes_data`.

### :gear: Environment Setup

It is recommend to train and test the model on linux, like ubuntu 20.04 with nvidia GPU. The CUDA compile tools with version 11.3 should be installed formly.

First, the python environment should be created through
```shell
conda create -n spconv python=3.8
```
Then install the dependence through pip:
```shell
pip install -r requirements.txt
```
Then compile the spconv operator:
```shell
bash build.sh
```
### :muscle: Training

Train the model on SemanticKITTI
```shell
cd SFCNet
python train_SemanticKITTI.py --log_dir <LOG> 
```
Train the model on nuScenes
```shell
cd SFCNet
python train_SemanticKITTI.py --log_dir <LOG> --dataset nuscenes_trainset_spp --config config_frust_nuscenes
```

### :sagittarius: Evaluation
Eval the model on SemanticKITTI (, suppose the model has been put in `SFCNet/logs/log_kitti/checkpoints/best.pt`)
```shell
cd SFCNet
python val_SemanticKITTI.py --log_dir logs/log_kitti
```
Eval the model on nuScenes (, suppose the model has been put in `SFCNet/logs/log_nuscenes/checkpoints/best.pt`)
```shell
cd SFCNet
python val_SemanticKITTI_nus.py --log_dir logs/log_nuscenes
```



### :paperclips: Reference

If you find our work useful, please cite us
```
@inproceedings{
    zheng2024spherical,
    title={Spherical Frustum Sparse Convolution Network for LiDAR Point Cloud Semantic Segmentation},
    author={Zheng, Yu and Wang, Guangming and Liu, Jiuming and Pollefeys, Marc and Wang, Hesheng},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
}
```

### :book: License

- Our project is licensed under the MIT and Apache license 2.0 (for the spherical frustum library) License - see the LICENSE and LICENSE_LIB files for details.
- The [spconv](https://github.com/traveller59/spconv/) project is licensed under the Apache license 2.0 License. Our project is modified from the `v1.1` branch of spconv.
- The [CUDPP](https://github.com/cudpp/cudpp) hash code is licensed under BSD License.
- The [SemanticKITTI](https://semantic-kitti.org/dataset.html#licence) dataset is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike](https://creativecommons.org/licenses/by-nc-sa/4.0/) License.
- The nuScenes dataset is licensed under [Creative Commons Attribution-Sharealike 4.0 International Public License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (CC BY-SA 4.0).

---

### :handshake:Acknowledgement

Our model training and testing architecture is mainly built based on [RandLA-Net-pytorch](https://github.com/tsunghan-wu/RandLA-Net-pytorch). The spherical frustum library is built based on [spconv](https://github.com/traveller59/spconv/). Many thanks to these open-sourced projects.

