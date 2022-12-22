import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class 2DEncoder(nn.Module):
    def __init__(self, final_layer,bottleneck):
        super().__init__()
        self.base=nn.Sequential(nn.Conv2d(3,32,3,stride=1,padding="same"),
        nn.ReLU(),
        nn.Conv2d(32,32,3,stride=1,padding="same"),
        nn.ReLU(),
        nn.Conv2d(32,64,3,stride=2,padding="same"),
        nn.ReLU(),
        nn.Conv2d(64,64,3,stride=1,padding="same"),
        nn.ReLU(),
        nn.Conv2d(64,64,3,stride=1,padding="same"),
        nn.ReLU(),
        nn.Conv2d(64,128,3,stride=2,padding="same"),
        nn.ReLU(),
        nn.Conv2d(128,128,3,stride=1,padding="same"),
        nn.ReLU(),
        nn.Conv2d(128,128,3,stride=1,padding="same"),
        nn.ReLU(),
        nn.Conv2d(128,256,3,stride=2,padding="same"),
        nn.ReLU(),
        nn.Conv2d(256,256,3,stride=1,padding="same"),
        nn.ReLU(),
        nn.Conv2d(256,256,3,stride=1,padding="same"),
        nn.ReLU(),
        nn.Conv2d(256,512,3,stride=2,padding="same"),
        nn.ReLU(),
        nn.Conv2d(512,512,3,stride=1,padding="same"),
        nn.ReLU(),
        nn.Conv2d(512,512,3,stride=1,padding="same"),
        nn.ReLU(),
        nn.Conv2d(512,512,3,stride=1,padding="same"),
        nn.ReLU(),
        nn.Conv2d(512,512,5,stride=2,padding="same"))
        
        if final_layer=="variational":
            self.mu=nn.Linear(512,bottleneck)
            self.std=nn.Linear(512,bottleneck)
            
        else:
            self.latent=nn.Linear(512,bottleneck)
            
    
        
    def forward(self,x,final_layer):
        x=self.base(x)
        if final_layer=="variational":
            x=self.latent(x)
            return x
        else:
            return self.mu(x),self.std(x)
        
    
"""
    A decoding network which maps points from the latent space back onto the data space.
"""


class 2DDecoder(nn.Module):
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
        super(Decoder, self).__init__()
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
        
class 2DToPointCloud(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        
        enc_result=self.encoder(x)
        result=self.decoder(enc_result)
        return result