import argparse
import os
import time

import torch
import torch.optim as optim

from model.model_3d_autoencoder import AutoEncoder
from utils.losses import ChamferLoss
from data.shapenet import ShapeNet


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="./data")
parser.add_argument("--npoints", type=int, default=2048)
parser.add_argument("--mpoints", type=int, default=2025)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-6)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--log_dir", type=str, default="./log")
args = parser.parse_args()

# prepare training and testing dataset
# train_dataset = ShapeNetPartDataset(root=args.root, npoints=args.npoints, split='train', classification=False, data_augmentation=True)
# test_dataset = ShapeNetPartDataset(root=args.root, npoints=args.npoints, split='test', classification=False, data_augmentation=True)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
def main(config):
    train_dataset = ShapeNet("train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,  # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=2,
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
    )

    test_dataset = ShapeNet("test")
    test_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,  # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=2,
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
    )

    valid_dataset = ShapeNet("valid")
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model
    autoencoder = AutoEncoder()
    autoencoder.to(device)

    # loss function
    cd_loss = ChamferLoss()
    # optimizer
    optimizer = optim.Adam(
        autoencoder.parameters(),
        lr=args.lr,
        betas=[0.9, 0.999],
        weight_decay=args.weight_decay,
    )

    batches = int(len(train_dataset) / args.batch_size + 0.5)

    min_cd_loss = 1e3
    best_epoch = -1

    print("\033[31mBegin Training...\033[0m")
    for epoch in range(1, args.epochs + 1):
        # training
        start = time.time()
        autoencoder.train()
        for i, data in enumerate(train_dataloader):
            point_clouds, _ = data
            point_clouds = point_clouds.permute(0, 2, 1)
            point_clouds = point_clouds.to(device)
            recons = autoencoder(point_clouds)
            ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))

            optimizer.zero_grad()
            ls.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch {}/{} with iteration {}/{}: CD loss is {}.".format(
                        epoch,
                        args.epochs,
                        i + 1,
                        batches,
                        ls.item() / len(point_clouds),
                    )
                )

        # evaluation
        autoencoder.eval()
        total_cd_loss = 0
        with torch.no_grad():
            for data in test_dataloader:
                point_clouds, _ = data
                point_clouds = point_clouds.permute(0, 2, 1)
                point_clouds = point_clouds.to(device)
                recons = autoencoder(point_clouds)
                ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
                total_cd_loss += ls.item()

        # calculate the mean cd loss
        mean_cd_loss = total_cd_loss / len(test_dataset)

        # records the best model and epoch
        if mean_cd_loss < min_cd_loss:
            min_cd_loss = mean_cd_loss
            best_epoch = epoch
            torch.save(
                autoencoder.state_dict(),
                os.path.join(args.log_dir, "model_lowest_cd_loss.pth"),
            )

        # save the model every 100 epochs
        if (epoch) % 100 == 0:
            torch.save(
                autoencoder.state_dict(),
                os.path.join(args.log_dir, "model_epoch_{}.pth".format(epoch)),
            )

        end = time.time()
        cost = end - start

        print(
            "\033[32mEpoch {}/{}: reconstructed Chamfer Distance is {}. Minimum cd loss is {} in epoch {}.\033[0m".format(
                epoch, args.epochs, mean_cd_loss, min_cd_loss, best_epoch
            )
        )
        print(
            "\033[31mCost {} minutes and {} seconds\033[0m".format(
                int(cost // 60), int(cost % 60)
            )
        )
