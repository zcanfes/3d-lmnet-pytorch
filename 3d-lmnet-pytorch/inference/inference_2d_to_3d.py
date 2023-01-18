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
from model.model_3d import PointCloudDecoder
from utils.visualization import visualize_pointcloud, visualize_image
from utils.losses import ChamferLoss
from utils.losses import DiversityLoss, SquaredEuclideanError, LeastAbsoluteError
import os
from data.shapenet import ShapeNet

def test(encoder, decoder, test_dataloader, device, config):
    if config["loss_criterion"] == "variational":

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
        loss_criterion.to(device)

    encoder.eval()
    autoencoder.eval()

    total_test_loss=0.
    index=-1
    for i, batch in enumerate(test_dataloader):
        # Move batch to device
        print("Batch",i)
        index+=1
        
        with torch.no_grad():
            ShapeNet.move_batch_to_device(batch, device)
            with open(config["2d_inference_gt_point"] + str(index)+ ".npy", "wb") as f:
                np.save(f, batch["point"].permute(0, 2, 1))
            with open(config["2d_inference_gt_img"] + str(index)+ ".npy", "wb") as f:
                np.save(f, batch["img"])
            
            if config["loss_criterion"] == "variational":
                
                mu, log_var = encoder(batch["img"][12][:,:,:128,:128])
                std = torch.sqrt(torch.exp(log_var))
                pred_latent = mu + torch.randn(std.size()) * std
                pred_pointcloud = autoencoder.decoder(pred_latent)
                batch_min=1e3
                for j in range(len(pred_pointcloud)):
                    loss,_=chamfer_distance(batch["point"].permute(0, 2, 1), pred_pointcloud[j].permute(0, 2, 1))
                    
                    distance = loss.detach().cpu()
                    if distance<batch_min:
                        batch_min=distance
                    
                    print("Chamfer distance for prediciton",j,":",distance)
                    with open(config["2d_inference_variational"] + str(index)+"_"+str(j) + ".npy", "wb") as f:
                        np.save(f, pred_pointcloud[i])

                total_test_loss+=batch_min
            else:
                pred_pointcloud = decoder(encoder(batch["img"][12][:,:,:128,:128]))
                loss,_=chamfer_distance(batch["point"].permute(0, 2, 1), pred_pointcloud.permute(0, 2, 1))
                distance = loss.detach().cpu()
                print("Chamfer distance for batch",i,":",distance)
                with open(config["2d_inference_normal"] + str(index) + ".npy", "wb") as f:
                    np.save(f, pred_pointcloud)

                total_test_loss+=distance

                print("Chamfer loss value for prediction:", distance)
    print("Total test chamfer distance:", total_test_loss)

def main(config):
    device = torch.device("cpu")
    if torch.cuda.is_available() and config["device"].startswith("cuda"):
        device = torch.device(config["device"])
        print("Using device:", config["device"])
    else:
        print("Using CPU")

    test_dataset = ShapeNet("test")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )

    encoder = ImageEncoder(config["final_layer"], config["bottleneck"])
    encoder = encoder.load_state_dict(
        torch.load(config["encoder_path"], map_location="cpu")
    )
    autoencoder=AutoEncoder(config["autoencoder_bottleneck"],config["autoencoder_hidden_size"],config["autoencoder_output_size"])
    
    autoencoder.load_state_dict(torch.load(config["3d_autoencoder_path"], map_location="cpu"))
    
    
    
    encoder.to(device)
    autoencoder.to(device)

    test(
        encoder,
        autoencoder,
        test_dataloader,
        device,
        config,
    )

