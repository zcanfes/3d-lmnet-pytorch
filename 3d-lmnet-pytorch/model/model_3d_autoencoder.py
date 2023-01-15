import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

"""
    A decoding network which maps points from the latent space back onto the data space.
"""


class Encoder(nn.Module):
    """
    Encoder Module
    """
    def __init__(
        self
    ):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 256, 1)
        self.conv5 = nn.Conv1d(256, 512, 1)
        
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)

       
        
    def forward(self, x):

        # five layer MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Local Max Pooling
        x = torch.max(x, dim=-1)[0]
        #print("tensor size after encoder " + str(x.size()))
        return x


class Decoder(nn.Module):
    """
    Decoder Module
    """
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
        print(" initial input " + str(input_size) + " initial hidden " + str(hidden_size) + " initial output " + str(output_size) )
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
        # print("\ninitial tensor size in decoder " + str(x.size()))
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
        # print("tensor size after decoder " + str(x.size())+ "\n")
        x = x.x = x.view(-1,3,2048) # This should be same with the input point cloud shape and be compatible with conv layer
        #print("tensor size after decoder " + str(x.size())+ "\n")
        return x

class AutoEncoder(nn.Module):
    def __init__(self,
        input_size,
        hidden_size,
        output_size,
        ):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(input_size, hidden_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
