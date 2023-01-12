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




class ChamferLoss(torch.autograd.Function):
    """
    Calculate chamfer, forward, backward distance between ground truth and predicted
    point clouds. They may or may not be scaled.
    """

    @staticmethod
    def forward(ctx, y_true, y_pred):
        ctx.save_for_backward(y_true, y_pred)

        dists_forward = torch.cdist(y_true, y_pred, p=2)
        dists_forward = torch.sqrt(dists_forward).mean()

        # print("dists forward shape:", dists_forward)

        chamfer_distance = 2 * dists_forward
        return chamfer_distance
    
    @staticmethod
    def backward(ctx):
        yy_true, yy_pred = ctx.saved_tensors

        dists_backward = torch.cdist(yy_true, yy_pred, p=2)
        dists_backward = torch.sqrt(dists_backward).mean()

        return dists_backward