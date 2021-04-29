# -*- coding: utf-8 -*-
# date: 2018-11-29 20:07
import torch.nn as nn

from .layer_norm import LayerNorm
from .functional import clones


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """

    def __init__(self, layer, n):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        i=0
        for layer in self.layers:
           
            x = layer(x, memory[-1]+memory[i], src_mask, tgt_mask)
            i=i+1
        return self.norm(x)
        #Decoder layer forward 
        #x: torch.Size([100, 12, 512])  
        #memory : torch.Size([100, 7, 512])  
        #src_mask : torch.Size([100, 1, 7])  #
        #tgt_mask : torch.Size([100, 12, 12])
