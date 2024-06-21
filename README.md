# 3D-LMNet - Unofficial PyTorch Implementation
This repository contains the **unofficial PyTorch implementation** of the paper [3D-LMNet: Latent Embedding Matching For Accurate and Diverse 3D Point Cloud Reconstruction From a Single Image](https://arxiv.org/abs/1807.07796) which is originally implemented in TensorFlow. The official TensorFlow implementation of the paper provided by the authors can be found [here](https://github.com/val-iisc/3d-lmnet).

To cite the original work of the authors you can use the following: 
```
@inproceedings{mandikal20183dlmnet,
 author = {Mandikal, Priyanka and Navaneet, K L and Agarwal, Mayank and Babu, R Venkatesh},
 booktitle = {Proceedings of the British Machine Vision Conference ({BMVC})},
 title = {{3D-LMNet}: Latent Embedding Matching for Accurate and Diverse 3D Point Cloud Reconstruction from a Single Image},
 year = {2018}
}
```

## 1. Data
3D-LMNet trains and validates their models on the ShapeNet dataset and we followed their instructions. The rendered images from the ShapeNet dataset provided by [3d-r2n2](https://github.com/chrischoy/3D-R2N2) is used. We use prepared the sampled points on the corresponding object meshes from ShapeNet to generate the ground truth point clouds. This dataset is prepared by the original authors of 3D-LMNet and we use the link they provided [here](https://github.com/val-iisc/3d-lmnet/blob/master/README.md). Moreover, we split the data using *create_split_files.py* using 60-20-20 convention. We created splits for the whole dataclass for 13 classes, splits for only 3 classes (chair, airplane, car) and for chair class only. Please note that, our splits for each of those cases consist of train, validation, and test whereas 3D-LMNet only provides train and validation. For their splits you can refer to: [ShapeNet train/val split file](https://drive.google.com/open?id=10FR-2Lbn55POB1y47MJ12euvobi6mgtc). 

You can download the data using the links below:

* Rendered Images zip(~12.3 GB): http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
* ShapeNet pointclouds zip (~2.8 GB): https://drive.google.com/open?id=1cfoe521iTgcB_7-g_98GYAqO553W8Y0g

**_You should only upload ShapeNet_pointclouds.zip dataset, whose link we shared above,_ to this directory: data/shapenet/**
* Our 3D-LMNET.ipynb covers unzipping [ShapeNet_pointclouds.zip](https://drive.google.com/open?id=1cfoe521iTgcB_7-g_98GYAqO553W8Y0g), downloading and unzipping [Rendered Images](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz) dataset. 
The **data** directory's file structure should look like this after executing dataset preparation cells on the 3D-LMNET.ipynb:

--data/ <br>
&nbsp;&nbsp;--ShapeNetRendering/<br>
&nbsp;&nbsp;--ShapeNet_pointclouds/<br>
&nbsp;&nbsp;--splits/<br>

## 2. Requirements

### 2.1. Tested Environment

* Linux
* Python 3.8
* CUDA 12.2 
* PyTorch 1.12.0
* PyTorch3D 0.7.2

### 2.2. Install Dependencies

First, clone the repository and create a Conda environment:

```bash
git clone https://github.com/zcanfes/3d-lmnet-pytorch.git
conda create --name 3d-lmnet-pytorch python=3.8. -y
conda activate 3d-lmnet-pytorch
```

Install PyTorch:

```bash
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Install PyTorch3D:

```bash
pip install fvcore
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1120/download.html
```

Install other dependencies (this may take some time):

```bash
pip install -r requirements.txt
```

## 3. Training
Train the 3D autoencoder and the 2D image encoders using the following instructions.

### 3.1. Command Line

Create the config files for each of your experiments. Some config examples are provided in the `config` folder. 

1. To train the autoencoder model:

```bash
python train.py --config ./config/train_3d_ae.yaml --experiment 3d
```

2. To train the image encoders: 

```bash
python train.py --config ./config/train_2d_l1.yaml --experiment 2d_to_3d
python train.py --config ./config/train_2d_l2.yaml --experiment 2d_to_3d
python train.py --config ./config/train_2d_div.yaml --experiment 2d_to_3d
```

3. For inference:

```bash
python test.py --config ./config/test_3d_ae.yaml --experiment 3d
python test.py --config ./config/test_2d_l1.yaml --experiment 2d_to_3d
python test.py --config ./config/test_2d_l2.yaml --experiment 2d_to_3d
python test.py --config ./config/test_2d_div.yaml --experiment 2d_to_3d
```

4. For visualization of the images and point clouds: 

```bash
python pc_visualization.py --config ./config/pc_visualization.yaml
```

### 3.2. Notebook

After downloading, the whole training process can be done using the `3D-LMNET.ipynb` file. You can use the config dictionary to change the experimental setup.

1. To train the autoencoder model run the cells under `3D Point Cloud Autoencoder Training`. 

**Important Note:** The autoencoder should be trained or a pre-trained autoencoder should be use for the training of the 2D image encoder. 

2. To train the 2D image encoder model, run the cells under the `2D Image Encoder Training` depending on which variant you want to train.

* For training the non-probabilistic version of the image encoder (Variant I) and with L1 loss, run the cells under the `Variant I - Latent Matching with L1 Loss`. 
* For training the non-probabilistic version of the image encoder (Variant I) and with L2 loss, run the cells under the `Variant I - Latent Matching with L2 Loss`. 
* For training the probabilistic version of the image encoder (Variant II), run the cells under the `Variant II - Probabilistic Latent Matching`. 

Here, you can change the `lambda` parameter to increase/decrease the weight of the diversity loss. 

## 4. Inference

The inference stage outputs the input images, ground truth point cloud, and the reconstructed (predicted) point clouds in `.npy`file format. 
To run the code in the `3D-LMNET.ipynb`, run the cells under the `Inference` depending on the variant you want to do inference on. 

For each trained model, you can use the corresponding Inference cells and obtain the results.

### 4.1. Visualization (Rendering) of Point Clouds

After obtaining the `.npy` point cloud and image files in the Inference, you can use run the cells under `Visualize Reconstructed Point Clouds` in the `3D-LMNET.ipynb` file. Here, [PyTorch3d ](https://pytorch3d.org/) is used for rendering. You can change the camera setup and many more settings using the PyTorch3d documentation. 

## 5. The Pre-trained models

You can download our pre-trained models using the links below:

* [3D point cloud autoencoder model](https://drive.google.com/file/d/1unq5OYW8WBhb-7ccFWGU2m1B0SaBlCjm/view?usp=sharing)
* [2D image encoder - Variant I with L1 Loss](https://drive.google.com/file/d/1YnNNJ15hwcGqDQHb-7t4HMC-TC1S-1fT/view?usp=share_link)
* [2D image encoder - Variant I with L2 Loss](https://drive.google.com/file/d/1uS13xA3k28jT5kxFHLVUveqm2_kD-5I8/view?usp=share_link)
* [2D image encoder - Variant II with diversity loss weight=5.5](https://drive.google.com/file/d/1b90RlaXwkl37V0Ue8MpT7vJTMfCM5-Uz/view?usp=share_link)
* [2D image encoder - Variant II with diversity loss weight=0.5](https://drive.google.com/file/d/1-u4awBG05Kzk2cnkT4v0G-oqj2_7e874/view?usp=share_link)
* [2D image encoder - Variant II with diversity loss weight=0.0](https://drive.google.com/file/d/1TZ40LANluZqIZ4utURnzJ-w8UqydirhE/view?usp=share_link)

After downloading the pre-trained models and the datasets, you can run the code using the `3D-LMNET.ipynb` file. You should run the cells under the `Imports and Setup` as well as the `Inference`. 

**Important Note:** After downloading the pre-trained models, create a folder called `3d-lmnet-pytorch/3d-lmnet-pytorch/trained_models/` and move the models to that directory before running the Inference code. 

After the inference is over, you can run the cells under the `Visualize Reconstructed Point Clouds` to visualize your results using PyTorch3d. 

## 6. Our Results

### 6.1. 3D Point Cloud Autoencoder Reconstructions

<p align="center">
<img src="https://user-images.githubusercontent.com/56366573/217518538-593b8e59-34b9-46ca-b894-29a7c038499d.png"  width="30%" height="30%">
</p>

### 6.2. Single-View Reconstructions with Variant I - L1 Loss

<p align="center">
<img src=https://user-images.githubusercontent.com/56366573/217519250-a96dfc93-a3fa-492b-9eb6-563e2f1dfec2.png   width="30%" height="30%">
</p>

### Single-View Reconstructions with Variant I - L2 Loss

<p align="center">
<img src=https://user-images.githubusercontent.com/56366573/217519400-8756bc4c-5483-4c5f-af11-d1ccf70e965f.png   width="30%" height="30%">
</p>

### 6.3. -View Reconstructions with Variant II - Diversity Loss Weight = 5.5

<p align="center">
<img src=https://user-images.githubusercontent.com/56366573/217519584-c4c69cb6-572d-4208-bb30-a5819b231e9f.png   width="30%" height="30%">
</p>

### 6.4. Single-View Reconstructions with Variant II - Different Weights for Diversity Loss


<p align="center">
<img src=https://user-images.githubusercontent.com/56366573/217520645-03efb7a2-473e-420c-88a0-40b53ea62991.png  width="50%" height="50%">
</p>


## 7. Acknowledgment

This PyTorch implementation is based on the [original TensorFlow implementation](https://github.com/val-iisc/3d-lmnet) of the paper [3D-LMNet: Latent Embedding Matching For Accurate and Diverse 3D Point Cloud Reconstruction From a Single Image](https://arxiv.org/abs/1807.07796). The original TensorFlow implementation is licensed under the [MIT License](#Original-License-of-the-TensorFlow-Implementation) below, which is also provided in the original TensorFlow repository (see [the original license](https://github.com/val-iisc/3d-lmnet/blob/master/LICENSE) for more details).

## 8. Original License of the TensorFlow Implementation

```
MIT License

Copyright (c) 2018 Video Analytics Lab -- IISc

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```


## 9. License for This Implementation

The MIT license for this repository can be found [here](https://github.com/zcanfes/3d-lmnet-pytorch/blob/main/LICENSE).


