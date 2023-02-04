import argparse
import os
import time

import torch
import pytorch3d
import torch.optim as optim
import numpy as np
from model.model_3d_autoencoder import AutoEncoder
#from utils.losses import ChamferLoss
from pytorch3d.loss import chamfer_distance
from data.shapenet import ShapeNet
from utils.point_cloud import show_point_cloud
from google.colab import files



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
        shuffle=True,  
        num_workers=config["num_workers"],
        pin_memory=True,  
        drop_last=True
    )

    test_dataset = ShapeNet("test")
    test_dataloader = torch.utils.data.DataLoader(
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

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model
    autoencoder = AutoEncoder(config["bottleneck"],config["hidden_size"],config["output_size"])
    #autoencoder.load_state_dict(torch.load(config["autoencoder"]))
    autoencoder.to(device)

    # loss function
    # chamfer_loss = ChamferLoss.apply()
    # optimizer
    optimizer = optim.Adam(
        autoencoder.parameters(),
        lr=config["lr"],
        betas=[0.9, 0.999],
        weight_decay=config["weight_decay"],
    )

    batches = int(len(train_dataset) / config["batch_size"])

    min_chamfer_loss = 1e3
    best_epoch = -1
    
    print("\033[31mBegin Training...\033[0m")
    for epoch in range(1, config["max_epochs"] + 1):
        # training
        start = time.time()
        autoencoder.train()
        #print(next(iter(train_dataloader)))
        for i, data in enumerate(train_dataloader):
            #print(i)
            ShapeNet.move_batch_to_device(data, device)

            optimizer.zero_grad()

            point_clouds = data["point"]
            # print("point_clouds shape:", point_clouds.shape)
            point_clouds = point_clouds.permute(0, 2, 1)
            point_clouds=point_clouds.type(torch.cuda.FloatTensor)
            #point_clouds = point_clouds.to(device)

            recons = autoencoder(point_clouds)

            # if i > 1600:
            #   print("iteration: ", i, "|| point clouds: ", point_clouds.shape, "|| reconstruction: ", recons.shape)
            #loss = ChamferLoss.apply(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
            
            loss,_=chamfer_distance(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
            loss.backward()

            optimizer.step()
            # print("loss value:", loss, " || loss shape: ", loss.shape)

            if (i + 1) % 100 == 0:
                print(
                    "Epoch {}/{} with iteration {}/{}: CD loss is {}.".format(
                        epoch,
                        config["max_epochs"],
                        i + 1,
                        batches,
                        loss.detach().cpu()
                        #loss.item() # / len(point_clouds),
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
                
                f_name = '/content/drive/MyDrive/3d-lmnet-pytorch/3d-lmnet-pytorch/imgs/gt/' + str(index) + '.npy'
                with open(f_name, 'wb') as f:
                    np.save(f, (point_clouds.cpu().numpy()))

                # print("point_clouds shape:", point_clouds.shape)
                point_clouds = point_clouds.permute(0, 2, 1)
                point_clouds=point_clouds.type(torch.cuda.FloatTensor)
                #point_clouds = point_clouds.to(device)
                #point_clouds = point_clouds.cpu().numpy()
                
                #show_point_cloud(point_clouds)

                recons = autoencoder(point_clouds)
                #show_point_cloud(recons)
                #recons = recons.cpu().numpy()
                f_name_recons = '/content/drive/MyDrive/3d-lmnet-pytorch/3d-lmnet-pytorch/imgs/pred/' + str(index) + '.npy'
                with open(f_name_recons, 'wb') as f:
                    np.save(f, (recons.permute(0, 2, 1).cpu().numpy()))
                #print(((recons.cpu().numpy())))

                #loss = ChamferLoss.apply(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
                loss,_=chamfer_distance(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
                total_chamfer_loss += loss.detach().cpu()
                #total_chamfer_loss += loss.item()

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
            #files.download(os.path.join(config["log_dir"], "model_lowest_chamfer_loss.pth"))

        # save the model every 100 epochs
        if (epoch) % 100 == 0:
            torch.save(
                autoencoder.state_dict(),
                os.path.join(config["log_dir"], "model_epoch_{}.pth".format(epoch)),
            )
            #files.download(os.path.join(config["log_dir"], "model_epoch_{}.pth".format(epoch)))
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
    #files.download(os.path.join(config["log_dir"], "model_autoencoder_final.pth"))
