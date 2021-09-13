# FESTA: Flow Estimation via Spatial-Temporal Attention for Scene Point Clouds

Created by Haiyan Wang*,  Jiahao Pang*,  Muhammad A. Lodhi,  Yingli Tian,  Dong Tian from <a href="https://www.interdigital.com/" target="_blank">InterDigital</a> and <a href="https://www.ccny.cuny.edu/" target="_blank">The City College of New York</a>
<p align="center">
    <img src="assets/pipeline.png" alt="framework" width="100%" />
</p>

## Introduction

This repository contains the official implementation for our <a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_FESTA_Flow_Estimation_via_Spatial-Temporal_Attention_for_Scene_Point_Clouds_CVPR_2021_paper.pdf" target="_blank">FESTA</a> paper, which aims at estimating scene flow from 3D point clouds. Our paper has been accepted at CVPR 2021 as an oral presentation. With the development of depth-sensing technologies, precise 3D measurements are available via point clouds which have sparked new research in 3D scene flow. Nevertheless, it remains challenging to extract scene flow from point clouds due to the sparsity and irregularity in typical point cloud sampling patterns. One major issue related to irregular sampling is identified as the randomness during point set abstraction/feature extraction—an elementary process in many flow estimation scenarios. A novel Spatial Abstraction with Attention (SA2) layer is accordingly proposed to alleviate the unstable abstraction problem. Moreover, a Temporal Abstraction with Attention (TA2) layer is proposed to rectify attention in the temporal domain, leading to benefits with motions scaled in a larger range.

## Installation

* Our work is based on Python 3.6, PyTorch 1.1.0 and CUDA 10.0, example commands to set up a virtual environment with anaconda are:
```bash
conda create -n festa python=3.6
conda activate festa
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
```

* Clone our repo to a folder, suppose its name is ``FESTA``. Under the ``path/to/FESTA`` folder, install the basic requirements with:
```bash
pip install -r requirements.txt
```

* Part of our implementation relies on the repo [flownet3d_pytorch](https://github.com/hyangwinter/flownet3d_pytorch/tree/a2e05c7827830217d2f48d87a664dcbeb130f652). In another folder out of ``path/to/FESTA``, run the following commands to clone [flownet3d_pytorch](https://github.com/hyangwinter/flownet3d_pytorch/tree/a2e05c7827830217d2f48d87a664dcbeb130f652) then checkout to the commit that we use:
```bash
git clone https://github.com/hyangwinter/flownet3d_pytorch.git
cd flownet3d_pytorch
git checkout a2e05c78
```

* Replace the lines ``279``, ``280`` and ``292`` in ``path/to/flownet3d_pytorch/lib/pointnet2_utils.py`` with the following three lines, respectively:
```python
        grouped_xyz_ori = grouping_operation(xyz_trans, idx)  # line 279
        grouped_xyz = grouped_xyz_ori - new_xyz.transpose(1, 2).unsqueeze(-1) # line 280
        return new_features, grouped_xyz_ori, idx # line 292
```

* Copy a few files from ``path/to/flownet3d_pytorch`` to ``path/to/FESTA`` with the following commands:
```bash
cp -r path/to/flownet3d_pytorch/lib path/to/FESTA/lib
cp path/to/flownet3d_pytorch/util.py path/to/FESTA/util.py
```

* Suppose CUDA 10.0 is installed in ``/usr/local/cuda-10.0``, run the following commands under ``path/to/FESTA`` for installation:
```bash
export CUDA_HOME="/usr/local/cuda-10.0"
cd lib
python setup.py install
cd ..
```
## Training & Testing

The processed ``FlyingThings3D`` dataset, as well as the ``KITTI Scene Flow`` dataset, can be obtained by following the instructions of the [FlowNet3D](https://github.com/xingyul/flownet3d) repo.

The configurations for training and testing are specified by ``train.yaml`` and ``test.yaml`` under ``path/to/FESTA/config/``. Please modify the dataset path (indicated by the option ``dataset_path``), as well as other options according to your need.

Supposed there are two GPUs available, under the folder ``path/to/FESTA``, run the following command to start training:
```bash
sh scripts/train.sh
```
Similarly, testing can be launched by
```bash
sh scripts/test.sh
```

## Citation

If you find our work useful for your research, please consider citing:

```
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Haiyan and Pang, Jiahao and Lodhi, Muhammad A. and Tian, Yingli and Tian, Dong},
    title     = {FESTA: Flow Estimation via Spatial-Temporal Attention for Scene Point Clouds},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {14173-14182}
}
```

## Related Projects

[flownet3d_pytorch](https://github.com/hyangwinter/flownet3d_pytorch/tree/a2e05c7827830217d2f48d87a664dcbeb130f652)

[FlowNet3D](https://github.com/xingyul/flownet3d) (This implementation of FlowNet3D is based on TensorFlow)

[HPLFlowNet](https://github.com/laoreja/HPLFlowNet)

[PointPWC](https://github.com/DylanWusee/PointPWC)

[MeteorNet](https://github.com/xingyul/meteornet)

<br />
<p align="center">
    <img src="assets/demo.png" alt="FESTA Demo" width="100%" /> 
</p>