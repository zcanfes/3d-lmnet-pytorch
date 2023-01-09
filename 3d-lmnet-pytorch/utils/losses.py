import torch
import torch.nn as nn
import numpy as np


class DiversityLoss:
    def __init__(self, alpha, penalty_angle):
        self.alpha = alpha
        self.penalty_angle = penalty_angle

    def __call__(self, azimuth_input, z_sigma):
        z_alpha = self.get_alpha(azimuth_input)
        L_reg = torch.mean((torch.mean(z_sigma, dim=-1) - z_alpha) ** 2)
        return L_reg

    def get_alpha(self, azimuth):
        penalty_angle = np.pi / 180.0 * self.penalty_angle

        return self.alpha * np.exp(-((azimuth - np.pi) ** 2 / (penalty_angle**2)))


class ChamferLoss:
    """
    Calculate chamfer, forward, backward distance between ground truth and predicted
    point clouds. They may or may not be scaled.
    """

    def __init__(self):
        self.history = {}

    def backward(self):
        return self.history["backward"]

    def forward(self, y_true, y_pred):
        dists_forward = torch.cdist(y_true, y_pred, p=2)
        dists_backward = torch.cdist(y_pred, y_true, p=2)

        dists_forward = torch.mean(torch.sqrt(dists_forward), dim=1)
        dists_backward = torch.mean(torch.sqrt(dists_backward), dim=1)
        chamfer_distance = dists_backward + dists_forward
        self.history["backward"] = dists_backward
        return dists_forward, dists_backward, chamfer_distance