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


def train(model_image, autoencoder, train_dataloader, val_dataloader, device, config):

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

    # best training loss for saving the model
    best_loss = float("inf")

    for epoch in range(config["max_epochs"]):
        train_loss_epoch=0.
        for i, batch in enumerate(train_dataloader):
            # Move batch to device
            ShapeNet.move_batch_to_device(batch, device)
            point_clouds = batch["point"]
            # print("point_clouds shape:", point_clouds.shape)
            point_clouds = point_clouds.permute(0, 2, 1)
            point_clouds=point_clouds.type(torch.cuda.FloatTensor)
            latent_from_pointcloud = autoencoder.encoder(point_clouds)
            
            
                
            image=batch["img"]
            image=image.type(torch.cuda.FloatTensor)
            
            # TODO: get azimuth angle
            AZIMUTH_INPUT = batch["azimuth"]
            #AZIMUTH_INPUT=AZIMUTH_INPUT.type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            
            if config["final_layer"]=="variational":
                mu, log_var = model_image(image)
                std = torch.sqrt(torch.exp(log_var))
                predicted_latent_from_2d = (mu + torch.randn(std.size()) * std).cuda()

                
            else:
                predicted_latent_from_2d=model_image(image)

            if config["loss_criterion"] == "variational":
                loss = loss_latent_matching(predicted_latent_from_2d, latent_from_pointcloud) + LAMBDA * loss_div(ALPHA, PENALTY_ANGLE, AZIMUTH_INPUT, predicted_latent_from_2d)
            else:
                loss = loss_criterion(predicted_latent_from_2d, latent_from_pointcloud)
               
            l=loss.item()
            loss.backward()
            optimizer.step()
            # loss logging
            
            
            train_loss_total += l
            train_loss_epoch += l

        print(f'[{epoch:03d}/{i:05d}] train loss: {train_loss_epoch}')
        

        # validation evaluation and logging
        if epoch % config["validate_every_n"] == 0:
            #loss=ChamferLoss()
            # set model to eval, important if your network has e.g. dropout or batchnorm layers
            model_image.eval()

            loss_total_val = 0.
            # forward pass and evaluation for entire validation set
            #index=-1
            for batch_val in val_dataloader:
                ShapeNet.move_batch_to_device(batch_val, device)

                with torch.no_grad():
                    point_clouds = batch_val["point"]
                    # print("point_clouds shape:", point_clouds.shape)
                    point_clouds = point_clouds.permute(0, 2, 1)
                    point_clouds=point_clouds.type(torch.cuda.FloatTensor)
                    latent_from_pointcloud = autoencoder.encoder(point_clouds)
                    index += 1
                    image=batch_val["img"]
                    image=image.type(torch.cuda.FloatTensor)
                    AZIMUTH_INPUT = batch_val["azimuth"]
                    AZIMUTH_INPUT=AZIMUTH_INPUT.type(torch.cuda.FloatTensor)
                    
                    if config["final_layer"]=="variational":
                        mu, log_var = model_image(image)
                        
                        # IMPLEMENT SAMPLING !!!!!!
                        std = torch.sqrt(torch.exp(log_var))
                        predicted_latent_from_2d = mu + torch.randn(std.size()) * std
                    else:
                        predicted_latent_from_2d=model_image(image)
                    """loss_total_val += loss(
                        prediction, model_pointcloud(batch_val["point"])
                    ).item()"""
                    
                    if config["loss_criterion"] == "variational":
                        loss_ = loss = loss_latent_matching(predicted_latent_from_2d, latent_from_pointcloud) + LAMBDA * loss_div(ALPHA, PENALTY_ANGLE, AZIMUTH_INPUT, predicted_latent_from_2d)
                    else:
                        loss_ = loss_criterion(predicted_latent_from_2d, latent_from_pointcloud)

                    loss_total_val += loss_.item()

            print("Validation loss:",loss_total_val)

            # TODO: calculate accuracy

            distance = loss_total_val
            if distance > best_distance:
                torch.save(
                    model_image.state_dict(),
                    os.path.join(config["experiment_name"],"model_best_epoch_{}.pth".format(epoch)),
                )
                
                best_distance = distance

            # set model back to train
            model_image.train()

        if epoch>0 and epoch%5==0:
          torch.save(
                model_image.state_dict(),
                os.path.join(config["experiment_name"],"model_epoch_{}.pth".format(epoch)),
            )

    torch.save(
        model_image.state_dict(),
        os.path.join(config["experiment_name"],"model_final.pth"),
    )

def main(config):
    """
    :param config: configuration for training - has the following keys
                   'experiment_name': name of the experiment, checkpoint will be saved to folder "3d-lmnet-pytorch/3d-lmnet-pytorch/runs/<experiment_name>"
                   'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                   'num_sample_points': number of sdf samples per shape while training
                   'bottleneck': length of the final latent vector
                   'batch_size': batch size for training and validation dataloaders
                   'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                   'learning_rate_model': learning rate of the encoder
                   'max_epochs': total number of epochs after which training should stop
                   'print_every_n': print train loss every n iterations
                   'ThreeDeeEncoderPath': path to the learned weights of ThreeDeeEncoder model
                   'visualize_every_n': visualize some training shapes every n iterations
                   'final_layer: if it is "variational" then mu and std are predicted or else a latent vector is predicted
                   'is_overfit': if the training is done on a small subset of data specified in 3d-lmnet-pytorch/3d-lmnet-pytorch/split/overfit.txt,
                                 train and validation done on the same set, so error close to 0 means a good overfit. Useful for debugging.
    """

    # declare device
    device = torch.device("cpu")
    if torch.cuda.is_available() and config["device"].startswith("cuda"):
        device = torch.device(config["device"])
        print("Using device:", config["device"])
    else:
        print("Using CPU")

    # create dataloaders
    train_dataset = ShapeNet("train",config["cat"],image_model=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,  # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=2,
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
    )

    val_dataset = ShapeNet("valid",config["cat"],image_model=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )

    # Instantiate model
    model_image = ImageEncoder(config["final_layer"], config["bottleneck"])

    # upload learned weights !!!!!!!!!
    autoencoder=AutoEncoder(config["autoencoder_bottleneck"],config["autoencoder_hidden_size"],config["autoencoder_output_size"])
    
    autoencoder.load_state_dict(torch.load(config["3d_encoder_path"], map_location="cpu"))
    

    # Load model if resuming from checkpoint
    if config["resume_ckpt"] is not None:
        model_image.load_state_dict(
            torch.load(config["resume_ckpt"] + "_model.pth", map_location="cpu")
        )

    # Move model to specified device
    model_image.to(device)
    autoencoder.to(device)
    # Create folder for saving checkpoints
    """Path(f'3d-lmnet-pytorch/3d-lmnet-pytorch/runs/{config["experiment_name"]}').mkdir(
        exist_ok=True, parents=True
    )"""

    # Start training
    train(
        model_image,
        autoencoder,
        train_dataloader,
        val_dataloader,
        device,
        config,
    )
