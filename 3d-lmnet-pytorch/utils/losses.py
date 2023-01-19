import torch
import torch.nn as nn
import numpy as np


class DiversityLoss(nn.Module):

    def __call__(self, alpha, penalty_angle, azimuth_input, z_sigma):
        z_alpha = self.get_alpha(alpha, penalty_angle, azimuth_input)
        #print("z_alpha")
        L_reg = torch.mean((torch.mean(z_sigma, dim=-1) - z_alpha.cuda()) ** 2)
        #print("l_reg")
        return L_reg

    def get_alpha(self, alpha, penalty_angle, azimuth):
        # print("get alpha")
        # penalty_angle = np.pi / 180.0 * penalty_angle
        # print("penalty")
        # expr= np.exp(-((azimuth - np.pi) ** 2))
        # expr = expr.detach().cpu()
        # print("expr")
        # alpha_expr = (alpha* expr**2).cpu()
        # print("alpha_expr")
        # penalty_expr =  (penalty_angle**2).cpu()
        # return (alpha_expr / penalty_expr)
        penalty_angle = np.pi / 180.0 * penalty_angle

        return alpha * np.exp(-((azimuth.cpu() - np.pi) ** 2 / (penalty_angle **2)))
