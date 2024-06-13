import os
import time
import torch
import pytorch3d
import torch.optim as optim
import numpy as np
from model.model_3d_autoencoder import AutoEncoder
from pytorch3d.loss import chamfer_distance
from data.shapenet import ShapeNet



def main(config):
    train_dataset = ShapeNet("train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,  
        num_workers=config["num_workers"],
        pin_memory=True,  
        drop_last=True
    )
    val_dataset = ShapeNet("valid")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"],
        drop_last=True
    )
    test_dataset = ShapeNet("test")

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model
    autoencoder = AutoEncoder(config["bottleneck"],config["hidden_size"],config["output_size"])
    autoencoder.to(device)

    print(config["lr"])

    # optimizer
    optimizer = optim.Adam(
        autoencoder.parameters(),
        lr=float(config["lr"]),
        betas=[0.9, 0.999],
        weight_decay=float(config["weight_decay"]),
    )

    batches = int(len(train_dataset) / float(config["batch_size"]))

    min_chamfer_loss = 1e3
    best_epoch = -1
    
    print("\033[31mTraining started...\033[0m")
    for epoch in range(1, config["max_epochs"] + 1):
        # training
        start = time.time()
        autoencoder.train()
        for i, data in enumerate(train_dataloader):
            ShapeNet.move_batch_to_device(data, device)

            optimizer.zero_grad()

            point_clouds = data["point"]
            point_clouds = point_clouds.permute(0, 2, 1)
            point_clouds=point_clouds.type(torch.cuda.FloatTensor)

            recons = autoencoder(point_clouds)
            
            loss,_=chamfer_distance(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
            loss.backward()

            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch {}/{} with iteration {}/{}: CD loss is {}.".format(
                        epoch,
                        config["max_epochs"],
                        i + 1,
                        batches,
                        loss.detach().cpu()
                    )
                )

        # evaluation
        autoencoder.eval()
        total_chamfer_loss = 0
        with torch.no_grad():
            index=-1
            for data in val_dataloader:
                index+=1
                ShapeNet.move_batch_to_device(data, device)

                optimizer.zero_grad()

                point_clouds = data["point"]

                point_clouds = point_clouds.permute(0, 2, 1)
                point_clouds=point_clouds.type(torch.cuda.FloatTensor)

                recons = autoencoder(point_clouds)

                loss,_=chamfer_distance(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
                total_chamfer_loss += loss.detach().cpu()

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
    torch.save(
          autoencoder.state_dict(),
          os.path.join(config["log_dir"], "model_autoencoder_final.pth"),
      )
