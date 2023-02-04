from pathlib import Path
import json

import numpy as np
import torch
import os
import cv2
from itertools import product

class ShapeNet(torch.utils.data.Dataset):
    num_classes = 13
    img_path = Path("/content/3d-lmnet-pytorch/3d-lmnet-pytorch/data/ShapeNetRendering")
    point_path = Path("/content/3d-lmnet-pytorch/3d-lmnet-pytorch/data/ShapeNet_pointclouds")

    num_to_class_mapping = json.loads(
        Path("/content/3d-lmnet-pytorch/3d-lmnet-pytorch/data/shape_info.json").read_text()
    )  # mapping for ShapeNet ids -> names
    class_to_nums_mapping=json.loads(Path("/content/3d-lmnet-pytorch/3d-lmnet-pytorch/data/class_to_nums.json").read_text())
    class_nums = sorted(num_to_class_mapping.keys())

    def __init__(self, split, cat=13,image_model=False):
        super().__init__()
        assert split in ["train", "valid", "test"]
        self.image_model=image_model
        if cat == 3:
            temp = (
                Path(
                    f"/content/3d-lmnet-pytorch/3d-lmnet-pytorch/data/splits/shapenet_point_cat3/{split}_cat3.txt"
                )
                .read_text()
                .splitlines()
            )  
            if self.image_model:
                idx=np.arange(24)
                self.point_items=list(product(temp,idx))
            else:
                self.point_items=temp

        elif cat == 1:
            temp = (
                Path(
                    f"/content/3d-lmnet-pytorch/3d-lmnet-pytorch/data/splits/shapenet_point_chair/{split}_chair.txt"
                )
                .read_text()
                .splitlines()
            )  
            if self.image_model:
                idx=np.arange(24)
                self.point_items=list(product(temp,idx))
            else:
                self.point_items=temp
        else:
            temp = (
                Path(f"/content/3d-lmnet-pytorch/3d-lmnet-pytorch/data/splits/shapenet_point/{split}.txt")
                .read_text()
                .splitlines()
            ) 
            if self.image_model:
                idx=np.arange(24)
                self.point_items=list(product(temp,idx))
            else:
                self.point_items=temp

        self.img_items = self.point_items

    def __getitem__(self, index):
        
        if not self.image_model:
            img_path= self.img_items[index]
            point_path = self.point_items[index]
            img,azimuth = ShapeNet.get_img_and_azimuth(img_path)
            point = ShapeNet.get_point_numpy(point_path)
        else:
            img_path,img_index = self.img_items[index]
            point_path,_ = self.point_items[index]
            img,azimuth = ShapeNet.get_img_and_azimuth(img_path,img_index)
            point = ShapeNet.get_point_numpy(point_path)
        return {"img": img, "point": point,"azimuth":azimuth}

    def __len__(self):
        
        return len(self.point_items)

    @staticmethod
    def move_batch_to_device(batch, device):
        # TODO add code to move batch to device
        batch["img"] = batch["img"].to(device)
        batch["point"] = batch["point"].to(device)
        batch["azimuth"]=batch["azimuth"].to(device)

    @staticmethod
    def get_point_numpy(shapenet_id):
        np_arr = np.load(ShapeNet.point_path / shapenet_id / "pointcloud_2048.npy")
        
        convertedArray = np_arr.astype(np.float)

        return convertedArray

    @staticmethod
    def get_img_and_azimuth(shapenet_id,img_index=0):
        np_res = []
        np_azimuth=[]
        meta_file=str(ShapeNet.img_path / shapenet_id / "rendering" / "rendering_metadata.txt")
        
        meta=np.loadtxt(meta_file)
                
        file_name=str(img_index).rjust(2, "0")+".png"
        p = str(ShapeNet.img_path / shapenet_id / "rendering" / file_name)
        image = cv2.imread(p)[4:-5, 4:-5, :3]
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        azimuth=(np.pi / 180.)*meta[img_index][0]
        return np.transpose(image,(2,0,1)),np.array(azimuth)

    
