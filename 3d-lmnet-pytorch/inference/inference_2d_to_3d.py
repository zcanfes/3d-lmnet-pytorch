import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import os
import pytorch3d
from pytorch3d.loss import chamfer_distance
from data.shapenet import ShapeNet
from model.model_2d import ImageEncoder
from model.model_3d_autoencoder import AutoEncoder
from utils.visualization import visualize_pointcloud, visualize_image
from utils.losses import DiversityLoss
import os
from data.shapenet import ShapeNet

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def test(encoder, autoencoder, test_dataloader, device, config,len_test_dataset):
    """if config["loss_criterion"] == "variational":
        # TODO: DiversityLoss TANIMLA !!!!!!!
        loss_div = DiversityLoss(config["alpha"], config["penalty_angle"])
        loss_latent_matching = nn.MSELoss()
        loss_latent_matching.to(device)
        loss_div.to(device)
    else:
        if config["loss_criterion"] == "L1":
            loss_criterion = nn.L1Loss()
        else:
            loss_criterion = nn.MSELoss()
        loss_criterion.to(device)"""

    encoder.eval()
    autoencoder.eval()

    total_test_loss=0.
    index=-1
    for i, batch in enumerate(test_dataloader):
        
        index+=1
        
        with torch.no_grad():
            ShapeNet.move_batch_to_device(batch, device)
            point_clouds = batch["point"]
            
            # print("point_clouds shape:", point_clouds.shape)
            point_clouds = point_clouds.permute(0, 2, 1)
            point_clouds=point_clouds.type(torch.cuda.FloatTensor)

            image=batch["img"]
            
            image=image.type(torch.cuda.FloatTensor)
            
            if i<50:
                with open(config["2d_inference_gt_point"] + "pointcloud_"+str(index)+ ".npy", "wb") as f:
                    np.save(f, point_clouds.permute(0, 2, 1).cpu().numpy())
                with open(config["2d_inference_gt_img"] + "image_"+str(index)+ ".npy", "wb") as f:
                    np.save(f, image.cpu().numpy())
            
            if config["loss_criterion"] == "variational":
                
                mu, log_var = encoder(image)
                std = torch.sqrt(torch.exp(log_var))
                
                pred_latent = mu + torch.randn((3,512),device=device) * std
                pred_pointcloud = autoencoder.decoder(pred_latent) 
                distance=0.
                for j in range(3):
                    loss,_=chamfer_distance(point_clouds.permute(0,2,1), pred_pointcloud[j,None,:,:].permute(0,2,1))
                    
                    distance += loss.detach().cpu()
                    
                    
                    #print("Chamfer distance for test data:",i," the",j,"th prediciton is :",distance)
                    if i<50:
                        with open(config["2d_inference_variational"] + "inference_"+str(index)+"_"+str(j) + ".npy", "wb") as f:
                            np.save(f, pred_pointcloud[j,None,:,:].cpu().numpy())
                total_test_loss+=distance/3
            else:
                enc_output=encoder(image)
                
                pred_pointcloud = autoencoder.decoder(enc_output)
                
                #print(point_clouds.size(),pred_pointcloud.size())
                loss,_=chamfer_distance(point_clouds.permute(0,2,1), pred_pointcloud.permute(0,2,1))
                distance = loss.detach().cpu()
                #print("Chamfer distance for test input",i,":",distance)
                if i<50:
                    with open(config["2d_inference_normal"] + "inference_"+str(index) + ".npy", "wb") as f:

                        np.save(f, pred_pointcloud.cpu().numpy())

                total_test_loss+=distance

    print("Total test chamfer distance:", total_test_loss/len(test_dataloader))

def main(config):
    device = torch.device("cpu")
    if torch.cuda.is_available() and config["device"].startswith("cuda"):
        device = torch.device(config["device"])
        print("Using device:", config["device"])
    else:
        print("Using CPU")

    test_dataset = ShapeNet("test",config["cat"],image_model=True)
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2, worker_init_fn=seed_worker,
    generator=g,
    )
    len_test_dataset=len(test_dataset)
    encoder = ImageEncoder(config["final_layer"], config["bottleneck"])
    encoder.load_state_dict(
        torch.load(config["encoder_path"], map_location="cpu")
    )
    autoencoder=AutoEncoder(config["autoencoder_bottleneck"],config["autoencoder_hidden_size"],config["autoencoder_output_size"])
    
    autoencoder.load_state_dict(torch.load(config["3d_autoencoder_path"], map_location="cpu"))
    
    print("Length of test datsaset:",len_test_dataset)
    
    encoder.to(device)
    autoencoder.to(device)

    test(
        encoder,
        autoencoder,
        test_dataloader,
        device,
        config,
        len_test_dataset
    )
