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
        self.conv1d7 = nn.Conv1d(7, 7, 3, 1, 1)
        self.conv1d12 = nn.Conv1d(12, 12, 3, 1, 1)


    def convo7(self,node):
        conv1=[]
        conv1.append(self.conv1d7(node[0]+node[1]))
        conv1.append(self.conv1d7(node[0]+node[1]+node[2]))
        conv1.append(self.conv1d7(node[2]+node[1]+node[3]))
        conv1.append(self.conv1d7(node[3]+node[2]+node[4]))
        conv1.append(self.conv1d7(node[4]+node[3]+node[5]))
        conv1.append(self.conv1d7(node[5]+node[4]+node[6]))
        conv1.append(self.conv1d7(node[6]+node[5]+node[7]))
        conv1.append(self.conv1d7(node[7]+node[6]))
        return conv1

    def convo12(self,node):
        conv1=[]
        conv1.append(self.conv1d12(node[0]+node[1]))
        conv1.append(self.conv1d12(node[0]+node[1]+node[2]))
        conv1.append(self.conv1d12(node[2]+node[1]+node[3]))
        conv1.append(self.conv1d12(node[3]+node[2]+node[4]))
        conv1.append(self.conv1d12(node[4]+node[3]+node[5]))
        conv1.append(self.conv1d12(node[5]+node[4]+node[6]))
        conv1.append(self.conv1d12(node[6]+node[5]+node[7]))
        conv1.append(self.conv1d12(node[7]+node[6]))
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
            nodekey.append(key[i])

        for i in range(0,self.h):
            nodequery.append(query[i])

        for i in range(0,self.h):
            nodevalue.append(value[i])



        #print(np.shape(nodequery))
        #print(np.shape(nodequery[0]))
        #print('0000000000000000000000000000000000000000000000000000000000000')
        
        if np.shape(nodekey[0])[1]==7:
            nodekey   = self.convo7(nodekey)
            nodevalue = self.convo7(nodevalue)
        if np.shape(nodekey[0])[1]==12:
            nodekey   = self.convo12(nodekey)
            nodevalue = self.convo12(nodevalue)

        if np.shape(nodequery[0])[1]==7:
            nodequery   = self.convo7(nodequery)

        if np.shape(nodequery[0])[1]==12:
            nodequery = self.convo12(nodequery)



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
        
        k = self.reunion(nodekey)
        q = self.reunion(nodequery)
        v = self.reunion(nodevalue)

        #print(np.shape(k))
        
        #print(np.shape(q))

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
