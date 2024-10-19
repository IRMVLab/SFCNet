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

<h4 align="center">We will clean up and release the codes soon!</h4>

### :newspaper:News

- **[26/Sept/2024]** Our Paper has been accepted as a Poster in **NeurIPS 2024**.

<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a>Abstract</a>
    </li>
          <li>
      <a>Reference</a>
    </li>
  </ol>
</details>

### :page_facing_up: Abstract

LiDAR point cloud semantic segmentation enables the robots to obtain fine-grained semantic information of the surrounding environment. Recently, many works project the point cloud onto the 2D image and adopt the 2D Convolutional Neural Networks (CNNs) or vision transformer for LiDAR point cloud semantic segmentation. However, since more than one point can be projected onto the same 2D position but only one point can be preserved, the previous 2D projection-based segmentation methods suffer from inevitable quantized information loss, which results in incomplete geometric structure, especially for small objects. To avoid quantized information loss, in this paper, we propose a novel spherical frustum structure, which preserves all points projected onto the same 2D position. Additionally, a hash-based representation is proposed for memory-efficient spherical frustum storage. Based on the spherical frustum structure, the Spherical Frustum sparse Convolution (SFC) and Frustum Farthest Point Sampling (F2PS) are proposed to convolve and sample the points stored in spherical frustums respectively. Finally, we present the Spherical Frustum sparse Convolution Network (SFCNet) to adopt 2D CNNs for LiDAR point cloud semantic segmentation without quantized information loss. Extensive experiments on the SemanticKITTI and nuScenes datasets demonstrate that our SFCNet outperforms previous 2D projection-based semantic segmentation methods based on conventional spherical projection and shows better performance on small object segmentation by preserving complete geometric structure. 

### :paperclips: Reference

```
@inproceedings{
zheng2024spherical,
title={Spherical Frustum Sparse Convolution Network for LiDAR Point Cloud Semantic Segmentation},
author={Zheng, Yu and Wang, Guangming and Liu, Jiuming and Pollefeys, Marc and Wang, Hesheng},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
}
```

