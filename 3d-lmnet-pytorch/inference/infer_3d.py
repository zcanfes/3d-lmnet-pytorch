import time
import torch
import pytorch3d
import torch.optim as optim
import numpy as np
from model.model_3d_autoencoder import AutoEncoder
from pytorch3d.loss import chamfer_distance
from data.shapenet import ShapeNet


def main(config):
    test_dataset = ShapeNet("test")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=True,  
        num_workers=config["num_workers"],
        pin_memory=True,  
        drop_last=True
    )


    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model
    autoencoder = AutoEncoder(config["bottleneck"],config["hidden_size"],config["output_size"])
    autoencoder.load_state_dict(torch.load(config["autoencoder"]))
    autoencoder.to(device)

    autoencoder.eval()
    print("\033[31mBegin Training...\033[0m")
    # training
    start = time.time()
    total_loss = 0
    index = -1
    print("Batch size is " + str(config["batch_size"]))
    for i, data in enumerate(test_dataloader):
        index+= 1
        with torch.no_grad():
            ShapeNet.move_batch_to_device(data, device)
            point_clouds = data["point"]
            point_clouds = point_clouds.permute(0, 2, 1)
            point_clouds=point_clouds.type(torch.cuda.FloatTensor)

            f_name_pcs = config["infer_gt"] + str(index) + '.npy'
            with open(f_name_pcs, 'wb') as f:
                np.save(f, (point_clouds.permute(0, 2, 1).cpu().numpy()))
                
            recons = autoencoder(point_clouds)
            f_name_recons = config["infer_pred"]  + str(index) + '.npy'
            with open(f_name_recons, 'wb') as f:
                np.save(f, (recons.permute(0, 2, 1).cpu().numpy()))

            loss,_=chamfer_distance(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
            loss_detach = loss.detach().cpu()
            total_loss += loss_detach

            
            print(
                    "{}. Input, CD loss is {}.".format(
                        i + 1,
                    
                        loss_detach
                        )
                )
        
    print("Total loss is: {}".format(total_loss/len(test_dataset)))
