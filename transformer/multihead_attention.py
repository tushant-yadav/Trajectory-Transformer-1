# -*- coding: utf-8 -*-
# date: 2018-11-30 16:35
import torch.nn as nn
import numpy as np
from .functional import clones, attention
import torch

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        #  We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.conv1d = nn.Conv1d(7, 7, 3, 1, 1)


    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        """
        node=[]
        conv1=[]
        splited =[]
        atn = []
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            nbatches = query.size(0)

        #spliting the K,Q,V values 
        query=torch.split(query,self.d_k,-1)
        key=torch.split(key,self.d_k,-1)
        value=torch.split(value,self.d_k,-1)

        print('++++++++++++++++++++++++++ ++++++')

        #INPUT LAYER
        for i in range(0,self.h):
            node.append(torch.cat((key[i],value[i],query[i]),-1))
        print('=========================================== nodes formed')
        print(np.shape(node[0]))

        #SET1 : CONV1D LAYERS
        conv1.append(self.conv1d(node[0]+node[1]))
        conv1.append(self.conv1d(node[0]+node[1]+node[2]))
        conv1.append(self.conv1d(node[2]+node[1]+node[3]))
        conv1.append(self.conv1d(node[3]+node[2]+node[4]))
        conv1.append(self.conv1d(node[4]+node[3]+node[5]))
        conv1.append(self.conv1d(node[5]+node[4]+node[6]))
        conv1.append(self.conv1d(node[6]+node[5]+node[7]))
        conv1.append(self.conv1d(node[7]+node[6]))
        print(np.shape(conv1[0]))
        print('********************************************* reaching to attn ')
        #attention
        for i in range(0,self.h):
            splited.append(torch.split(conv1[i],self.d_k,-1)) 
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ splited')
        print(np.shape(splited[0][0]))
        for i in range(0,self.h):
            atn[i] = attention(splited[i][2], splited[i][0], splited[i][1], mask=mask, dropout=self.dropout)
        print('attention going+++++++++++++++++++++++++++++++++++++++')
        x, self.attn = torch.cat((atn[0],atn[1],atn[2],atn[3],atn[4],atn[5],atn[6],atn[7]),-1)
        print('attention gained &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        return self.linears[-1](x)
