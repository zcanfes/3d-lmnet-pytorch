import torch
import torch.nn as nn
import numpy as np

class DiversityLoss(nn.Module):

    def __call__(self, alpha, penalty_angle, azimuth_input, z_sigma):
        z_alpha = self.get_alpha(alpha, penalty_angle, azimuth_input)
        L_reg = torch.mean((torch.mean(z_sigma, axis=-1) - z_alpha)**2)
        return L_reg

    def get_alpha(self, alpha, penalty_angle, azimuth):
        penalty_angle = (np.pi / 180.0) * penalty_angle
        return (alpha * np.exp(-((azimuth.cpu()-np.pi)**2/(penalty_angle**2)))).cuda()
