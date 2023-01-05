import torch
import torch.nn as nn


class DiversityLoss:
    pass


class ChamferLoss:
    """
    Calculate chamfer, forward, backward distance between ground truth and predicted
    point clouds. They may or may not be scaled.
    """

    def __call__(self, y_true, y_pred):
        dists_forward, _ = torch.cdist(y_true, y_pred, p=2)
        dists_backward, _ = torch.cdist(y_pred, y_true, p=2)

        dists_forward = torch.mean(torch.sqrt(dists_forward), dim=1)  # (B, )
        dists_backward = torch.mean(torch.sqrt(dists_backward), dim=1)  # (B, )
        chamfer_distance = dists_backward + dists_forward
        return dists_forward, dists_backward, chamfer_distance
