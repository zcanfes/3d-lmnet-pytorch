from pathlib import Path
import json

import numpy as np
import torch
import os
import cv2
class ShapeNet(torch.utils.data.Dataset):
    num_classes = 13
    img_path = Path("/content/term_project/data/ShapeNetRendering")  
    point_path=Path("/content/term_project/data/ShapeNet_pointclouds")
    
    class_name_mapping = json.loads(Path("/content/term_project/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split,cat=13):
        super().__init__()
        assert split in ['train', 'valid', 'test']
       # self.truncation_distance = 3
        if cat==3:
            self.point_items = Path(f"/content/term_project/data/splits/shapenet_point_cat3/{split}_cat3.txt").read_text().splitlines()  # keep track of shapes based on split
            
        elif cat==1:
            self.point_items = Path(f"/content/term_project/data/splits/shapenet_point_chair/{split}_chair.txt").read_text().splitlines()  # keep track of shapes based on split
        
        else:
            self.point_items = Path(f"/content/term_project/data/splits/shapenet_point/{split}.txt").read_text().splitlines()  # keep track of shapes based on split
            
        self.img_items=self.point_items

    def __getitem__(self, index):
        img_index= self.img_items[index]
        point_index=self.point_items[index]
        
        img= ShapeNet.get_img_numpy(img_index)
        point=ShapeNet.get_point_numpy(point_index)
        

        
        return {
            "img":img,
            "point":point
        }

    def __len__(self):
        return len(self.point_items)

    @staticmethod
    def move_batch_to_device(batch, device):
        # TODO add code to move batch to device
        batch['img'] = batch['img'].to(device)
        batch["point"]=batch["point"].to(device)
        
    @staticmethod
    def get_point_numpy(shapenet_id):
        np_arr = np.load(ShapeNet.point_path / shapenet_id /'pointcloud_2048.npy')
            
        return np_arr

    @staticmethod
    def get_img_numpy(shapenet_id):
        np_res=[]
        for _,_,files in os.walk(ShapeNet.img_path / shapenet_id):
            for f in files:
                if f[-3:]=="png":
                    p=str(ShapeNet.img_path / shapenet_id / "rendering" /f)
                    image=np.transpose(np.array(cv2.imread(p)),(1,2,0))
                    np_res.append(image)
        
            
        return np.array(np_res)
    
