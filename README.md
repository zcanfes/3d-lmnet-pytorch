# Implementation of 3D-LMNet with PyTorch
This repository contains the unofficial PyTorch implementation of the paper [3D-LMNet: Latent Embedding Matching For Accurate and Diverse 3D Point Cloud Reconstruction From a Single Image](https://arxiv.org/abs/1807.07796).

```
@inproceedings{mandikal20183dlmnet,
 author = {Mandikal, Priyanka and Navaneet, K L and Agarwal, Mayank and Babu, R Venkatesh},
 booktitle = {Proceedings of the British Machine Vision Conference ({BMVC})},
 title = {{3D-LMNet}: Latent Embedding Matching for Accurate and Diverse 3D Point Cloud Reconstruction from a Single Image},
 year = {2018}
}
```

## ShapeNet Dataset
3D-LMNet trains and validates their models on the ShapeNet dataset and we followed their instructions. The rendered images from the ShapeNet dataset provided by [3d-r2n2](https://github.com/chrischoy/3D-R2N2) is used. We use prepared the sampled points on the corresponding object meshes from ShapeNet to generate the ground truth point clouds. This dataset is prepared by the original authors of 3D-LMNet and we use the link they provided [here](https://github.com/val-iisc/3d-lmnet/blob/master/README.md). Moreover, we split the data using *create_split_files.py* using 60-20-20 convention. We created splits for the whole dataclass for 13 classes, splits for only 3 classes (chair, airplane, car) and for chair class only. Please note that, our splits for each of those cases consist of train, validation, and test whereas 3D-LMNet only provides train and validation. For their splits you can refer to: [ShapeNet train/val split file](https://drive.google.com/open?id=10FR-2Lbn55POB1y47MJ12euvobi6mgtc). 

You can download the data using the links below:

* Rendered Images (~12.3 GB): http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
* ShapeNet pointclouds (~2.8 GB): https://drive.google.com/open?id=1cfoe521iTgcB_7-g_98GYAqO553W8Y0g

After downloading, extract the folders and move them into data/shapenet/.
The folder structure should be:

--data/shapenet/ <br>
&nbsp;&nbsp;--ShapeNetRendering/<br>
&nbsp;&nbsp;--ShapeNet_pointclouds/<br>
&nbsp;&nbsp;--splits/<br>

## Run the code

### Use the pre-trained models

You can download our pre-trained models using the links below:

* [3D point cloud autoencoder model](https://drive.google.com/file/d/1unq5OYW8WBhb-7ccFWGU2m1B0SaBlCjm/view?usp=sharing)
* [2D image encoder - Variant I with L1 Loss](https://drive.google.com/file/d/1YnNNJ15hwcGqDQHb-7t4HMC-TC1S-1fT/view?usp=share_link)
* [2D image encoder - Variant I with L2 Loss](https://drive.google.com/file/d/1uS13xA3k28jT5kxFHLVUveqm2_kD-5I8/view?usp=share_link)
* [2D image encoder - Variant II with diversity loss weight=5.5](https://drive.google.com/file/d/1b90RlaXwkl37V0Ue8MpT7vJTMfCM5-Uz/view?usp=share_link)
* [2D image encoder - Variant II with diversity loss weight=0.5](https://drive.google.com/file/d/1-u4awBG05Kzk2cnkT4v0G-oqj2_7e874/view?usp=share_link)
* [2D image encoder - Variant II with diversity loss weight=0.0](https://drive.google.com/file/d/1TZ40LANluZqIZ4utURnzJ-w8UqydirhE/view?usp=share_link)


