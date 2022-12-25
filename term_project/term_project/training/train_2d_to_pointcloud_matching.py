from pathlib import Path

import torch

from term_project.model.model_2d import TwoDeeEncoder
from term_project.model.model_pointcloud import ThreeDeeEncoder

from term_project.data.shapenet import ShapeNet

# from exercise_3.util.misc import evaluate_model_on_grid

from google.colab import files
import os


def train(
    model2d, model_pointcloud, train_dataloader, valid_dataloader, device, config
):

    loss = None
    if config["loss_criterion"] == "variational":

        # loss_diversity TANIMLA !!!!!!!

        loss_latent_matching = nn.MSELoss()
        loss = loss_latent_matching + loss_diversity

        optimizer = torch.optim.Adam(
            [
                {
                    "params": model2d.base.parameters(),
                    "lr": config["learning_rate_model"],
                    "weight_decay": 1e-5,
                },
                {
                    "params": model2d.mu.parameters(),
                    "lr": config["learning_rate_model"],
                    "weight_decay": 1e-3,
                },
                {
                    "params": model2d.std.parameters(),
                    "lr": config["learning_rate_model"],
                    "weight_decay": 1e-3,
                },
            ]
        )
    else:
        if config["loss_criterion"] == "L1":
            loss = nn.L1Loss()
        else:
            loss = nn.MSELoss()

        optimizer = torch.optim.Adam(
            [
                {
                    "params": model2d.base.parameters(),
                    "lr": config["learning_rate_model"],
                    "weight_decay": 1e-5,
                },
                {
                    "params": model2d.latent.parameters(),
                    "lr": config["learning_rate_model"],
                    "weight_decay": 1e-3,
                },
            ]
        )
    loss.to(device)
    model2d.train()
    model_pointcloud.eval()

    train_loss_running = 0.0

    # best training loss for saving the model
    best_loss = float("inf")

    for epoch in range(config["max_epochs"]):

        for i, batch in enumerate(train_dataloader):
            # Move batch to device
            ShapeNet.move_batch_to_device(batch, device)

            optimizer.zero_grad()

            mu, log_var = model2d(batch["img"][12])
            #  IMLEMENT SAMPLING !!!!!!!
            std = torch.sqrt(torch.exp(log_var))
            predicted_latent_from_2d = mu + torch.randn(std.size()) * std

            latent_from_pointcloud = model_pointcloud(batch["point"])
            loss = loss_criterion(predicted_latent_from_2d, latent_from_pointcloud)

            # TODO: backward
            loss.backward()

            # TODO: update network parameters
            optimizer.step()

            # loss logging
            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + i

            if iteration % config["print_every_n"] == (config["print_every_n"] - 1):
                print(
                    f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}'
                )
                train_loss_running = 0.0

            # validation evaluation and logging
            if iteration % config["validate_every_n"] == (
                config["validate_every_n"] - 1
            ):

                # set model to eval, important if your network has e.g. dropout or batchnorm layers
                model2d.eval()

                loss_total_val = 0
                total, correct = 0, 0
                # forward pass and evaluation for entire validation set
                for batch_val in valid_dataloader:
                    ShapeNetVox.move_batch_to_device(batch_val, device)

                    with torch.no_grad():
                        # TODO: Get prediction scores
                        mu, log_var = model2d(batch_val["img"][12])

                        # IMPLEMENT SAMPLING !!!!!!
                        std = torch.sqrt(torch.exp(log_var))
                        prediction = mu + torch.randn(std.size()) * std

                    loss_total_val += loss(
                        prediction, model_pointcloud(batch_val["point"])
                    ).item()

                print(
                    f"[{epoch:03d}/{i:05d}] val_loss: {loss_total_val / len(valloader):.3f}"
                )

                if accuracy > best_accuracy:
                    torch.save(
                        model.state_dict(),
                        f'term_project/runs/{config["experiment_name"]}/model_best.ckpt',
                    )
                    best_accuracy = accuracy

                # set model back to train
                model2d.train()


def main(config):
    """
    Function for training DeepSDF
    :param config: configuration for training - has the following keys
                   'experiment_name': name of the experiment, checkpoint will be saved to folder "term_project/runs/<experiment_name>"
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
                   'is_overfit': if the training is done on a small subset of data specified in term_project/split/overfit.txt,
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
    train_dataset = ShapeNet("train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,  # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=2,
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
    )

    valset = ShapeNet("valid")
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )

    # Instantiate model
    model2d = TwoDeeEncoder(config["final_layer"], config["bottleneck"])

    # upload learned weights !!!!!!!!!
    model_pointcloud = ThreeDeeEncoder.load_state_dict(
        torch.load(config["ThreeDeeEncoderPath"], map_location="cpu")
    )

    # Load model if resuming from checkpoint
    if config["resume_ckpt"] is not None:
        model2d.load_state_dict(
            torch.load(config["resume_ckpt"] + "_model.ckpt", map_location="cpu")
        )

    # Move model to specified device
    model2d.to(device)
    model_pointcloud.to(device)
    # Create folder for saving checkpoints
    Path(f'term_project/runs/{config["experiment_name"]}').mkdir(
        exist_ok=True, parents=True
    )

    # Start training
    train(model2d, model_pointcloud, train_dataloader, valid_dataloader, device, config)
