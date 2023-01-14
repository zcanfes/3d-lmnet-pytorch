import random
from pathlib import Path
import numpy as np
import torch
import os

from model.model_3d_autoencoder import Encoder, Decoder
from utils.visualization import visualize_pointcloud, visualize_image
from utils.losses import ChamferLoss


class Inference3DToPointCloudVariational:
    def __init__(self, inp, encoder_path, decoder_path, config, device):
        encoder = Encoder()
        self.encoder = encoder.load_state_dict(
            torch.load(encoder_path, map_location="cpu")
        )
        decoder = Decoder(
            config["input_size"],
            config["hidden_size"],
            config["output_size"],
        )
        self.decoder = decoder.load_state_dict(
            torch.load(decoder_path, map_location="cpu")
        )
        self.input = inp
        self.device = device
        self.encoder.eval()
        self.encoder.eval()
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def infer(self):
        loss_criterion = ChamferLoss()
        loss_criterion.to(self.device)

        mu, log_var = self.encoder(self.input["img"][12])
        std = torch.sqrt(torch.exp(log_var))
        pred_latent = mu + torch.randn(std.size()) * std
        pred_pointcloud = self.decoder(pred_latent)

        loss_value = []
        print("Groundtruth 3D image:")
        visualize_image(self.input["img"])

        print("Groundtruth point cloud:")
        visualize_pointcloud(self.input["point"])

        p = (
            "./generated_variatioal_pointclouds_from3d_image/"
            + str(self.pointcloud_filename)
            + "/"
        )
        isExist = os.path.exists(p)
        if not isExist:

            os.makedirs(p)

        for i in range(len(pred_pointcloud)):
            _, _, distance = loss_criterion(pred_pointcloud[i], self.input["point"]).item()
            loss_value.append(
                distance
            )

            print("Chamfer loss value for prediction", i, ":", loss_value)

            print("Prediction:")
            visualize_pointcloud(pred_pointcloud[i])

            with open(p + str(self.pointcloud_filename) + ".npy", "wb") as f:
                np.save(f, pred_pointcloud)
                self.pointcloud_filename += 1


class Inference3DToPointCloudNormal:
    def __init__(self, inp, encoder_path, decoder_path, config, device):
        encoder = Encoder()
        self.encoder = encoder.load_state_dict(
            torch.load(encoder_path, map_location="cpu")
        )
        decoder = Decoder(
            config["input_size"],
            config["hidden_size"],
            config["output_size"],
        )
        self.decoder = decoder.load_state_dict(
            torch.load(decoder_path, map_location="cpu")
        )
        self.input = inp
        self.device = device
        self.encoder.eval()
        self.encoder.eval()
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.pointcloud_filename = 0

    def infer(self):

        # TODO: CHAMFER LOSS SHOULD BE DEFINED !!!!!!!!!!!
        loss_criterion = ChamferLoss()
        loss_criterion.to(self.device)

        pred_pointcloud = self.decoder(self.encoder(self.input["img"][12]))
        print("Groundtruth 3D image:")
        visualize_image(self.input["img"])

        print("Groundtruth pointcloud:")
        visualize_pointcloud(self.input["point"])
        _, _, loss_value = loss_criterion(pred_pointcloud, self.input["point"]).item()

        print("Chamfer loss value for prediction:", loss_value)

        print("Prediction pointcloud:")
        visualize_pointcloud(pred_pointcloud)

        p = (
            "./generated_pointcloud_from3d_image/"
            + str(self.pointcloud_filename)
            + "/"
        )
        isExist = os.path.exists(p)
        if not isExist:

            os.makedirs(p)

        with open(p + str(self.pointcloud_filename) + ".npy", "wb") as f:
            np.save(f, pred_pointcloud)
            self.pointcloud_filename += 1