import random
from pathlib import Path

import torch

from term_project.data.shapenet import ShapeNet
from term_project.model.model_2d import TwoDeeEncoder
from term_project.model.model_3d import ThreeDeeDecoder
from term_project.util.visualization import visualize_pointcloud


class Inference2DToPointcloudVariational:
    def __init__(self, inp, encoder_path, decoder_path, config, device):
        encoder = TwoDeeEncoder(config["final_layer"], config["bottleneck"])
        self.encoder = encoder.load_state_dict(
            torch.load(encoder_path, map_location="cpu")
        )
        decoder = ThreeDeeDecoder(
            config["input_size"],
            config["hidden_size"],
            config["output_size"],
            config["bnorm"],
            config["bnorm_final"],
            config["regularizer"],
            config["weight_decay"],
            config["dropout_prob"],
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

        # CHAMFER LOSS SHOULD BE DEFINED !!!!!!!!!!!
        loss = ...
        loss.to(device)

        mu, log_var = self.encoder(self.input["img"][12])
        std = torch.sqrt(torch.exp(log_var))
        pred_latent = mu + torch.randn(std.size()) * std
        pred_pointcloud = self.decoder(pred_latent)

        loss_value = []
        print("Groundtruth:")
        visualize_pointcloud(self.input["point"])
        for i in range(len(pred_pointcloud)):
            loss_value.append(loss(pred_pointcloud[i], self.input["point"]).item())

            print("Chamfer loss value for prediction", i, ":", loss_value)

            print("Prediction:")
            visualize_pointcloud(pred_pointcloud[i])


class Inference2DToPointcloudNormal:
    def __init__(self, inp, encoder_path, decoder_path, config, device):
        encoder = TwoDeeEncoder(config["final_layer"], config["bottleneck"])
        self.encoder = encoder.load_state_dict(
            torch.load(encoder_path, map_location="cpu")
        )
        decoder = ThreeDeeDecoder(
            config["input_size"],
            config["hidden_size"],
            config["output_size"],
            config["bnorm"],
            config["bnorm_final"],
            config["regularizer"],
            config["weight_decay"],
            config["dropout_prob"],
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

        # CHAMFER LOSS SHOULD BE DEFINED !!!!!!!!!!!
        loss = ...
        loss.to(device)

        pred_pointcloud = self.decoder(self.encoder(self.input["img"][12]))
        print("Groundtruth:")
        visualize_pointcloud(self.input["point"])
        loss_value = loss(pred_pointcloud, self.input["point"]).item()

        print("Chamfer loss value for prediction:", loss_value)

        print("Prediction:")
        visualize_pointcloud(pred_pointcloud)
