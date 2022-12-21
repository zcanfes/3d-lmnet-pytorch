import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
    A decoding network which maps points from the latent space back onto the data space.
"""


class Decoder(nn.Module):
    def __init__(
        self,
        latent_signal,
        layer_sizes=[],
        b_norm=True,
        non_linearity=F.relu,
        regularizer=None,
        weight_decay=0.001,
        reuse=False,
        scope=None,
        dropout_prob=None,
        b_norm_finish=False,
        verbose=False,
    ):
        super(Decoder, self).__init__()
        self.latent_signal = latent_signal
        self.layer_sizes = layer_sizes
        self.b_norm = b_norm
        self.non_linearity = non_linearity
        self.regularization = regularizer
        self.weight_decay = weight_decay
        self.b_norm_finish = b_norm_finish
        self.verbose = verbose
        self.layers = []

        if self.verbose:
            print("Building decoder...")

        n_layers = len(layer_sizes)

        # replicate the drop_out prob for all layers -> instead of tf_utils.replicate_parameter_for_all_layers
        if dropout_prob is not None:
            dropout_prob = [dropout_prob] * n_layers
        self.dropout_prob = dropout_prob

        if n_layers < 2: # if only one layer choose a simpler architecture
            raise ValueError("For an FC decoder with single a layer use simpler code.")
        
        # create the layers
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            name =  f"decoder_fc_{i}"
            
            if i == 0:
                layer = latent_signal # first layer is the latent signal
            else:
                layer = nn.Linear(layer_sizes[i-1], layer_sizes[i]) # create a linear layer
            
            self.layers.append(layer)

            if self.verbose: 
                print(name, f"FC params = {np.prod(layer.weight.shape) + np.prod(layer.bias.shape)}")
            
            if b_norm: 
                name += "_bnorm"
                layer = nn.BatchNorm1d(layer_sizes[i])
                self.layers.append(layer)
            
                if self.verbose:
                    print(f"bnorm params = {np.prod(layer.weight.shape) + np.prod(layer.bias.shape)}")

            if dropout_prob is not None:
                layer = nn.Dropout(dropout_prob[i])
                self.layers.append(layer)
            
            if non_linearity is not None:
                layer = non_linearity(layer)
                self.layers.append(layer)

            if self.verbose:
                print(f"Decoder layer {i} output size: {np.prod(layer.shape)[1:]}")

        # final layer is a linear layer

        name = f"decoder_fc_{n_layers-1}"
        layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.layers.append(layer)

        if self.verbose:
            print(name, f"FC params = {np.prod(layer.weight.shape) + np.prod(layer.bias.shape)}")

        if self.b_norm_finish:
            name += "_bnorm"
            layer = nn.BatchNorm1d(layer_sizes[-1])
            self.layers.append(layer)

            if self.verbose:
                print(f"bnorm params = {np.prod(layer.weight.shape) + np.prod(layer.bias.shape)}")
            
        if self.verbose:
            print("Decoder output size: ", np.prod(layer.shape)[1:])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x




            
        
            
