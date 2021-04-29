# -*- coding: utf-8 -*-
# date: 2018-11-29 20:07
import torch.nn as nn

from .functional import clones
from .layer_norm import LayerNorm


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, x_mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        x_list = []
    
        for layer in self.layers:
            x = layer(x, x_mask)
            x_list.append(x)
        
        for i in range(0,len(x_list)):
            x_list[i] = layer(x_list[i], x_mask)
        return x_list
