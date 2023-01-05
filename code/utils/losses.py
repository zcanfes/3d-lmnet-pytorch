import torch
import torch.nn as nn
import numpy as np


class DiversityLoss:
    def get_alpha(self, azimuth):
        penalty_angle = np.pi / 180.0 * self.penalty_angle

        return self.alpha * np.exp(-((azimuth - np.pi) ** 2 / (penalty_angle**2)))

    def __call__(self, azimuth_input, z_sigma):
        z_alpha = self.get_alpha(azimuth_input)
        L_reg = torch.mean((torch.mean(z_sigma, dim=-1) - z_alpha) ** 2)
        return L_reg

    def __init__(self, alpha, penalty_angle):
        self.alpha = alpha
        self.penalty_angle = penalty_angle


class ChamferLoss:
    """
    Calculate chamfer, forward, backward distance between ground truth and predicted
    point clouds. They may or may not be scaled.
    """

    def __call__(self, y_true, y_pred):
        dists_forward, _ = torch.cdist(y_true, y_pred, p=2)
        dists_backward, _ = torch.cdist(y_pred, y_true, p=2)

        dists_forward = torch.mean(torch.sqrt(dists_forward), dim=1)
        dists_backward = torch.mean(torch.sqrt(dists_backward), dim=1)
        chamfer_distance = dists_backward + dists_forward
        return dists_forward, dists_backward, chamfer_distance


class SquaredEuclideanError:
    """
    Calculate L2 loss between ground truth and predicted latent presentations.
    """

    def __call__(self, y_true, y_pred):
        l2 = torch.mean((y_true - y_pred) ** 2)
        return l2


class LeastAbsoluteError:
    """
    Calculate L1 loss between ground truth and predicted latent presentations.
    """

    def __call__(self, y_true, y_pred):
        l1 = torch.mean(torch.abs(y_true - y_pred))
        return l1
