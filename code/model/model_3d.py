import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

"""
    A decoding network which maps points from the latent space back onto the data space.
"""


class PointCloudDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        bnorm=True,
        bnorm_final=False,
        regularizer=None,
        weight_decay=0.001,
        dropout_prob=None,
    ):
        super().__init__()
        self.bnorm = bnorm
        self.bnorm_final = bnorm_final
        self.dropout_prob = dropout_prob

        if self.bnorm:
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        if self.bnorm_final:
            self.bn3 = nn.BatchNorm1d(output_size)

        if dropout_prob is not None:
            self.dropout = nn.Dropout(dropout_prob)

        self.relu = F.relu

    def forward(self, x):
        x = self.fc1(x)
        if self.bnorm:
            x = self.bn1(x)

        if self.dropout_prob is not None:
            x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        if self.bnorm:
            x = self.bn2(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)
        if self.bnorm_final:
            x = self.bn3(x)

        return x
