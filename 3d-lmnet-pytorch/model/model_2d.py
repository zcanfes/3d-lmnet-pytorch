import torch.nn as nn
import torch


class ImageEncoder(nn.Module):
    def __init__(self, final_layer, bottleneck):
        super().__init__()
        self.base = nn.Sequential(
            self.conv_block(3, 32, 3, stride=1),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            self.conv_block(64, 64, 3, stride=1),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            self.conv_block(128, 128, 3, stride=1),
            nn.Conv2d(128, 256, 3, stride=2),
            nn.ReLU(),
            self.conv_block(256, 256, 3, stride=1),
            nn.Conv2d(256, 512, 3, stride=2),
            nn.ReLU(),
            self.conv_block(512, 512, 3, stride=1),
            nn.Conv2d(512, 512, 3, stride=1,padding="same"),
            nn.ReLU(),
            nn.Conv2d(512, 512, 5, stride=2),
            nn.ReLU()
        )
        self.final_layer=final_layer
        self.mu=None
        self.std=None
        self.latent=None
        if self.final_layer == "variational":
            self.mu = nn.Linear(512*4, bottleneck)
            self.std = nn.Linear(512*4, bottleneck)

        else:
            
            self.latent = nn.Linear(512*4, bottleneck)

    def conv_block(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding="same"
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size, stride=stride, padding="same"
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.base(x)
        x=torch.flatten(x,start_dim=1)
        if self.final_layer == "variational":
            
            return self.mu(x),self.std(x)
        else:
            x = self.latent(x)
            return x
            
