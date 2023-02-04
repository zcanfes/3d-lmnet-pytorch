from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from model.model_2d import ImageEncoder
from model.model_3d_autoencoder import Encoder
from utils.losses import DiversityLoss
import os
from data.shapenet import ShapeNet
from model.model_3d_autoencoder import AutoEncoder


def train(model_image, autoencoder, train_dataloader, val_dataloader, device, config,len_train_dataset,len_val_dataset):

    loss_criterion = None
    best_distance = 1e3

    if config["loss_criterion"] == "variational":

        ALPHA = config["alpha"]
        PENALTY_ANGLE = config["penalty_angle"]
        LAMBDA = config["lambda"]

        

        loss_div = DiversityLoss()
        loss_latent_matching = nn.MSELoss()

        loss_latent_matching.to(device)
        loss_div.to(device)

        optimizer = torch.optim.Adam(
            [
                {
                    "params": model_image.base.parameters(),
                    "lr": config["learning_rate_model"],
                    "weight_decay": 1e-5,
                },
                {
                    "params": model_image.mu.parameters(),
                    "lr": config["learning_rate_model"],
                    "weight_decay": 1e-3,
                },
                {
                    "params": model_image.std.parameters(),
                    "lr": config["learning_rate_model"],
                    "weight_decay": 1e-3,
                },
            ]
        )

    else:

        if config["loss_criterion"] == "L1":
            loss_criterion = nn.L1Loss()
        else:
            loss_criterion = nn.MSELoss()

        loss_criterion.to(device)
        optimizer = torch.optim.Adam(
            [
                {
                    "params": model_image.base.parameters(),
                    "lr": config["learning_rate_model"],
                    "weight_decay": 1e-5,
                },
                {
                    "params": model_image.latent.parameters(),
                    "lr": config["learning_rate_model"],
                    "weight_decay": 1e-3,
                },
            ]
        )
    model_image.train()
    autoencoder.eval()

    train_loss_total = 0.0
    val_loss_total=0.
    count_val=0
    batches = int(len_train_dataset / config["batch_size"])

    # best training loss for saving the model
    best_loss = float("inf")

    for epoch in range(1,config["max_epochs"]+1):
        train_loss_epoch=0.
        for i, batch in enumerate(train_dataloader):
            ShapeNet.move_batch_to_device(batch, device)
            point_clouds = batch["point"]
            point_clouds = point_clouds.permute(0, 2, 1)
            point_clouds=point_clouds.type(torch.cuda.FloatTensor)
            latent_from_pointcloud = autoencoder.encoder(point_clouds)
            
            
                
            image=batch["img"]
            image=image.type(torch.cuda.FloatTensor)
            
            AZIMUTH_INPUT = batch["azimuth"]
            AZIMUTH_INPUT=AZIMUTH_INPUT.type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            if config["final_layer"]=="variational":
                mu, log_var = model_image(image)
                std = torch.sqrt(torch.exp(log_var))
                rand = torch.randn(std.size(),  device=device)
                predicted_latent_from_2d = (mu + rand * std).cuda()

            
            else:
                predicted_latent_from_2d=model_image(image)
           
            if config["loss_criterion"] == "variational":
                loss = loss_latent_matching(predicted_latent_from_2d, latent_from_pointcloud) + LAMBDA * loss_div(ALPHA, PENALTY_ANGLE, AZIMUTH_INPUT, predicted_latent_from_2d)
               
            else:
                loss = loss_criterion(predicted_latent_from_2d, latent_from_pointcloud)
            
            l=loss.item()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print(
                    "Epoch {}/{} with iteration {}/{}: Loss is {}.".format(
                        epoch,
                        config["max_epochs"],
                        i + 1,
                        batches,
                        l
                        #loss.item() # / len(point_clouds),
                    )
                )
            train_loss_total += l
            train_loss_epoch += l

        print(f'[{epoch:03d}/{i:05d}] train loss: {train_loss_epoch/len(train_dataloader)}',"Total train loss so far:",train_loss_total/(epoch*len(train_dataloader)))

        # validation evaluation and logging
        if epoch % config["validate_every_n"] == 0 and epoch>0:
           
            print("Validations starts...")
            model_image.eval()
            count_val+=1
            loss_val = 0.
            
            for batch_val in val_dataloader:
                ShapeNet.move_batch_to_device(batch_val, device)

                with torch.no_grad():
                    point_clouds = batch_val["point"]
                    point_clouds = point_clouds.permute(0, 2, 1)
                    point_clouds=point_clouds.type(torch.cuda.FloatTensor)
                    latent_from_pointcloud = autoencoder.encoder(point_clouds)
                    optimizer.zero_grad()
                    image=batch_val["img"]
                    image=image.type(torch.cuda.FloatTensor)
                    AZIMUTH_INPUT = batch_val["azimuth"]
                    AZIMUTH_INPUT=AZIMUTH_INPUT.type(torch.cuda.FloatTensor)
                    
                    if config["final_layer"]=="variational":
                        mu, log_var = model_image(image)
                        
                        std = torch.sqrt(torch.exp(log_var))
                        rand = torch.randn(std.size(),  device=device)
                        predicted_latent_from_2d = (mu + rand * std).cuda()
                        
                    else:
                        predicted_latent_from_2d=model_image(image)
                   
                    
                    if config["loss_criterion"] == "variational":
                        loss = loss_latent_matching(predicted_latent_from_2d, latent_from_pointcloud) + LAMBDA * loss_div(ALPHA, PENALTY_ANGLE, AZIMUTH_INPUT, predicted_latent_from_2d)
                    else:
                        loss = loss_criterion(predicted_latent_from_2d, latent_from_pointcloud)

                    loss_val += loss.item()
                    val_loss_total+=loss.item()
            print("Validation loss:",loss_val/len(val_dataloader),"Total validation loss so far:",val_loss_total/(count_val*len(val_dataloader)))

            
            distance = loss_val
            if distance > best_distance:
                torch.save(
                    model_image.state_dict(),
                    os.path.join(config["experiment_name"],"model_best_epoch_{}.pth".format(epoch)),
                )
                
                best_distance = distance

            model_image.train()

        if epoch%config["save_every_n"]==0:
          torch.save(
                model_image.state_dict(),
                os.path.join(config["experiment_name"],"model_epoch_{}.pth".format(epoch)),
            )
    print("Total training loss:",train_loss_total/(config["max_epochs"]*len(train_dataloader)))
    print("Total validation loss:",val_loss_total/(count_val*len(val_dataloader)))

    torch.save(
        model_image.state_dict(),
        os.path.join(config["experiment_name"],"model_final.pth"),
    )


def main(config):
    device = torch.device("cpu")
    if torch.cuda.is_available() and config["device"].startswith("cuda"):
        device = torch.device(config["device"])
        print("Using device:", config["device"])
    else:
        print("Using CPU")

    train_dataset = ShapeNet("train",config["cat"],image_model=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True, 
        num_workers=2,
        pin_memory=True,  
    )

    val_dataset = ShapeNet("valid",config["cat"],image_model=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )
    model_image = ImageEncoder(config["final_layer"], config["bottleneck"])

    autoencoder=AutoEncoder(config["autoencoder_bottleneck"],config["autoencoder_hidden_size"],config["autoencoder_output_size"])
    
    autoencoder.load_state_dict(torch.load(config["3d_encoder_path"], map_location="cpu"))
    

    if config["resume_ckpt"] is not None:
        model_image.load_state_dict(
            torch.load(config["resume_ckpt"], map_location="cpu")
        )
    model_image.to(device)
    autoencoder.to(device)
   
    
    train(
        model_image,
        autoencoder,
        train_dataloader,
        val_dataloader,
        device,
        config,
        len(train_dataset),
        len(val_dataset)
    )
