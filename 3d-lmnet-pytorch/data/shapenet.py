from pathlib import Path
import json

import numpy as np
import torch
import os
import cv2


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 13
    img_path = Path("/content/3d-lmnet-pytorch/3d-lmnet-pytorch/data/ShapeNetRendering")
    point_path = Path("/content/3d-lmnet-pytorch/3d-lmnet-pytorch/data/ShapeNet_pointclouds")

    num_to_class_mapping = json.loads(
        Path("/content/3d-lmnet-pytorch/3d-lmnet-pytorch/data/shape_info.json").read_text()
    )  # mapping for ShapeNet ids -> names
    class_to_nums_mapping=json.loads(Path("/content/3d-lmnet-pytorch/3d-lmnet-pytorch/data/class_to_nums.json").read_text())
    class_nums = sorted(num_to_class_mapping.keys())

    def __init__(self, split, cat=13):
        super().__init__()
        assert split in ["train", "valid", "test"]
        # self.truncation_distance = 3
        if cat == 3:
            self.point_items = (
                Path(
                    f"/content/3d-lmnet-pytorch/3d-lmnet-pytorch/data/splits/shapenet_point_cat3/{split}_cat3.txt"
                )
                .read_text()
                .splitlines()
            )  # keep track of shapes based on split

        elif cat == 1:
            self.point_items = (
                Path(
                    f"/content/3d-lmnet-pytorch/3d-lmnet-pytorch/data/splits/shapenet_point_chair/{split}_chair.txt"
                )
                .read_text()
                .splitlines()
            )  # keep track of shapes based on split

        else:
            self.point_items = (
                Path(f"/content/3d-lmnet-pytorch/3d-lmnet-pytorch/data/splits/shapenet_point/{split}.txt")
                .read_text()
                .splitlines()
            )  # keep track of shapes based on split

        self.img_items = self.point_items

    def __getitem__(self, index):
        img_index = self.img_items[index]
        point_index = self.point_items[index]

        img,azimuth = ShapeNet.get_img_and_azimuth(img_index)
        point = ShapeNet.get_point_numpy(point_index)

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
    def get_img_and_azimuth(shapenet_id):
        np_res = []
        np_azimuth=[]
        meta_file=str(ShapeNet.img_path / shapenet_id / "rendering" / "rendering_metadata.txt")
        
        meta=np.loadtxt(meta_file)
        for _, _, files in os.walk(ShapeNet.img_path / shapenet_id):
            png_files=[int(f[:-4]) for f in files if f[-3:] == "png"]
            
            png_files.sort()
            for i,f in enumerate(png_files):
                file_name=str(f).rjust(2, "0")+".png"
                p = str(ShapeNet.img_path / shapenet_id / "rendering" / file_name)

                image = cv2.imread(p)[4:-5, 4:-5, :3]
                image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                np_res.append(image)
                np_azimuth.append((np.pi / 180.)*meta[i][0])
        return np.array(np_res),np.array(np_azimuth)

    
