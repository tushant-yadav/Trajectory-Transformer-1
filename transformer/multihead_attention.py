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
        self.conv1d = nn.Conv1d(64, 64, 3, 1, 1)


    def convo(self,node):
        conv1=[]
        conv1.append(self.conv1d(node[0]+node[1]))
        conv1.append(self.conv1d(node[0]+node[1]+node[2]))
        conv1.append(self.conv1d(node[2]+node[1]+node[3]))
        conv1.append(self.conv1d(node[3]+node[2]+node[4]))
        conv1.append(self.conv1d(node[4]+node[3]+node[5]))
        conv1.append(self.conv1d(node[5]+node[4]+node[6]))
        conv1.append(self.conv1d(node[6]+node[5]+node[7]))
        conv1.append(self.conv1d(node[7]+node[6]))
        return conv1  

    def reunion(self, matrix ):
        k = torch.unsqueeze(matrix[0],1)
        for i in range(1,8):
            k = torch.cat((k,torch.unsqueeze(matrix[i],1)),1)

        return k

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        """
        #print('multihead entry:',np.shape(query))
        #print('multihead entry:',np.shape(key))
        #print('multihead entry:',np.shape(value))
        nodekey=[]
        nodequery=[]
        nodevalue=[]
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

        #print('++++++++++++++++++++++++++ ++++++')
        #print(np.shape(query))
        #print(np.shape(query[0]))
        #INPUT LAYER




        for i in range(0,self.h):
            nodekey.append(torch.transpose(key[i],1,2))

        for i in range(0,self.h):
            nodequery.append(torch.transpose(query[i],1,2))

        for i in range(0,self.h):
            nodevalue.append(torch.transpose(value[i],1,2))

        #print(np.shape(nodequery))
        #print(np.shape(nodequery[0]))
        #print('0000000000000000000000000000000000000000000000000000000000000')
        nodekey   = self.convo(nodekey)
        nodequery = self.convo(nodequery)
        nodevalue = self.convo(nodevalue)



        #print('=========================================== nodes formed')
        #print(np.shape(nodequery))
        #print(np.shape(nodequery[0]))

        #SET1 : CONV1D LAYERS

        #print(np.shape())
        #print('********************************************* reaching to attn ')
        #attention
        '''for i in range(0,self.h):
            splited.append(torch.split(torch.transpose(conv1[i],1,2),self.d_k,-1))
        '''
        
        k = torch.transpose(self.reunion(nodekey),2,3)
        q = torch.transpose(self.reunion(nodequery),2,3)
        v = torch.transpose(self.reunion(nodevalue),2,3)

        #print(np.shape(k))
        #print(np.shape(q))
        #print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ splited')
        #print(np.shape(splited))
        x, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        #print('x',np.shape(x))
        #print('self.attn:' ,np.shape(self.attn))
        #print('attention going+++++++++++++++++++++++++++++++++++++++')
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        #print('x1:',np.shape(x))
        return self.linears[-1](x)


        '''Query  torch.Size([100, 8, 12, 64])
        key  torch.Size([100, 8, 7, 64])
        value torch.Size([100, 8, 7, 64])
        Scores torch.Size([100, 8, 12, 7])
        Query  torch.Size([100, 8, 12, 64])
        key  torch.Size([100, 8, 12, 64])
        value torch.Size([100, 8, 12, 64])
        Scores torch.Size([100, 8, 12, 12])
        Query  torch.Size([100, 8, 12, 64])
        key  torch.Size([100, 8, 7, 64])
        value torch.Size([100, 8, 7, 64])
        Scores torch.Size([100, 8, 12, 7])
        Query  torch.Size([100, 8, 12, 64])
        key  torch.Size([100, 8, 12, 64])
        value torch.Size([100, 8, 12, 64])
        Scores torch.Size([100, 8, 12, 12])
        Query  torch.Size([100, 8, 12, 64])
        key  torch.Size([100, 8, 7, 64])
        value torch.Size([100, 8, 7, 64])
        Scores torch.Size([100, 8, 12, 7])'''
