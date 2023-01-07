import argparse
import os
import time

import torch
import torch.optim as optim

from model.model_3d_autoencoder import AutoEncoder
from utils.losses import ChamferLoss
from data.shapenet import ShapeNet




# prepare training and testing dataset
# train_dataset = ShapeNetPartDataset(root=args.root, npoints=args.npoints, split='train', classification=False, data_augmentation=True)
# test_dataset = ShapeNetPartDataset(root=args.root, npoints=args.npoints, split='test', classification=False, data_augmentation=True)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
def main(config):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--root", type=str, default="./data")
    # parser.add_argument("--npoints", type=int, default=2048)
    # parser.add_argument("--mpoints", type=int, default=2025)
    # parser.add_argument("--batch_size", type=int, default=16)
    # parser.add_argument("--lr", type=float, default=1e-4)
    # parser.add_argument("--weight_decay", type=float, default=1e-6)
    # parser.add_argument("--epochs", type=int, default=400)
    # parser.add_argument("--num_workers", type=int, default=4)
    # parser.add_argument("--log_dir", type=str, default="./log")
    # args = parser.parse_args()
    train_dataset = ShapeNet("train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,  # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=config["num_workers"],
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
    )

    test_dataset = ShapeNet("test")
    test_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,  # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=config["num_workers"],
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
    )

    val_dataset = ShapeNet("valid")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"]
    )

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model
    autoencoder = AutoEncoder(config["input_size"],config["hidden_size"],config["output_size"])
    autoencoder.to(device)

    # loss function
    chamfer_loss = ChamferLoss()
    # optimizer
    optimizer = optim.Adam(
        autoencoder.parameters(),
        lr=config["lr"],
        betas=[0.9, 0.999],
        weight_decay=config["weight_decay"],
    )

    batches = int(len(train_dataset) / config["batch_size"] + 0.5)

    min_chamfer_loss = 1e3
    best_epoch = -1

    print("\033[31mBegin Training...\033[0m")
    for epoch in range(1, config["max_epochs"] + 1):
        # training
        start = time.time()
        autoencoder.train()
        for i, data in enumerate(train_dataloader):
            point_clouds, _ = data
            print(point_clouds)
            point_clouds = point_clouds.to(device)
            recons = autoencoder(point_clouds)
            loss = chamfer_loss(point_clouds, recons)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch {}/{} with iteration {}/{}: CD loss is {}.".format(
                        epoch,
                        config["max_epochs"],
                        i + 1,
                        batches,
                        loss.item() / len(point_clouds),
                    )
                )

        # evaluation
        autoencoder.eval()
        total_chamfer_loss = 0
        with torch.no_grad():
            for data in test_dataloader:
                point_clouds, _ = data
                point_clouds = point_clouds.to(device)
                recons = autoencoder(point_clouds)
                _, _, loss = chamfer_loss(
                    point_clouds, recons
                )
                total_chamfer_loss += loss.item()

        # calculate the mean cd loss
        mean_chamfer_loss = total_chamfer_loss / len(test_dataset)

        # records the best model and epoch
        if mean_chamfer_loss < min_chamfer_loss:
            min_chamfer_loss = mean_chamfer_loss
            best_epoch = epoch
            torch.save(
                autoencoder.state_dict(),
                os.path.join(config["log_dir"], "model_lowest_chamfer_loss.pth"),
            )

        # save the model every 100 epochs
        if (epoch) % 100 == 0:
            torch.save(
                autoencoder.state_dict(),
                os.path.join(config["log_dir"], "model_epoch_{}.pth".format(epoch)),
            )

        end = time.time()
        cost = end - start

        print(
            "\033[32mEpoch {}/{}: reconstructed Chamfer Distance is {}. Minimum Chamfer loss is {} in epoch {}.\033[0m".format(
                epoch, config["max_epochs"], mean_chamfer_loss, min_chamfer_loss, best_epoch
            )
        )
        print(
            "\033[31mCost {} minutes and {} seconds\033[0m".format(
                int(cost // 60), int(cost % 60)
            )
        )