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
        self.conv1d12 = nn.Conv1d(12, 12, 3, 1, 1)

    def convo12(self,node,maskk,shape_k):
        conv1 = []
        conv2 = []
        conv3 = []
        conv1.append(self.conv1d12(node[0]+node[1]))
        conv1.append(self.conv1d12(node[0]+node[1]+node[2]))
        conv1.append(self.conv1d12(node[2]+node[1]+node[3]))
        conv1.append(self.conv1d12(node[3]+node[2]+node[4]))
        conv1.append(self.conv1d12(node[4]+node[3]+node[5]))
        conv1.append(self.conv1d12(node[5]+node[4]+node[6]))
        conv1.append(self.conv1d12(node[6]+node[5]+node[7]))
        conv1.append(self.conv1d12(node[7]+node[6]))



        conv2.append(self.conv1d12(torch.cat((conv1[0],conv1[1]),2)))
        conv2.append(self.conv1d12(torch.cat((conv1[2],conv1[3]),2)))
        conv2.append(self.conv1d12(torch.cat((conv1[4],conv1[5]),2)))
        conv2.append(self.conv1d12(torch.cat((conv1[6],conv1[7]),2)))



        conv3.append(self.conv1d12(torch.cat((conv2[0],conv2[1]),2)))
        conv3.append(self.conv1d12(torch.cat((conv2[2],conv2[2]),2)))


        for i in range(0,8):
            conv1[i] = torch.split(conv1[i],shape_k,1)[0]
        for i in range(0,4):
            conv2[i] = torch.split(conv2[i],shape_k,1)[0]
        for i in range(0,2):
            conv3[i] = torch.split(conv3[i],shape_k,1)[0]

        return conv1,conv2,conv3


    def reunion(self, matrix1, matrix2, matrix3 ):
        k1 = torch.unsqueeze(matrix1[0],1)
        for i in range(1,8):
            k1 = torch.cat((k1,torch.unsqueeze(matrix1[i],1)),1)

        k2 = torch.unsqueeze(matrix2[0],1)
        for i in range(1,4):
            k2 = torch.cat((k2,torch.unsqueeze(matrix2[i],1)),1)

        k3 = torch.unsqueeze(matrix3[0],1)
        for i in range(1,2):
            k3 = torch.cat((k3,torch.unsqueeze(matrix3[i],1)),1)

        return k1, k2, k3 

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
        flag = 0

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

        if torch.cuda.is_available():  
          dev = "cuda:0" 
        else:  
          dev = "cpu"  

        device = torch.device(dev)  


        shape_k = np.shape(nodekey[0])
        shape_q = np.shape(nodequery[0])
        maskk = torch.tensor([[[1e-9]*(12-shape_k[1])]*64]*shape_k[0])
        maskk = torch.transpose(maskk,1,2)
        maskq = torch.tensor([[[1e-9]*(12-shape_q[1])]*64]*shape_q[0])
        maskq = torch.transpose(maskq,1,2)
        maskk = maskk.to(device)
        maskq = maskq.to(device)

        #print(np.shape(nodequery))
        #print(np.shape(nodequery[0]))
        #print('0000000000000000000000000000000000000000000000000000000000000')
 
        for i in range(0,8):
            #print(np.shape(nodekey[i]),np.shape(maskk))
            #print('key cat:',i,np.shape(torch.cat((nodekey[i],maskk),1)))
            nodekey[i] = torch.cat((nodekey[i],maskk),1)
            #print(i,np.shape(nodekey[i]),np.shape(mask7) )
            #print('val cat:',i,np.shape(torch.cat((nodevalue[i],maskk),1)))

            nodevalue[i] = torch.cat((nodevalue[i],maskk),1)
            #print('que cat:',i,np.shape(torch.cat((nodequery[i],maskq),1)))

            nodequery[i] = torch.cat((nodequery[i],maskq),1)
 

        #print('input to convo',np.shape(nodekey[0]),np.shape(nodevalue[0]),np.shape(nodequery[0]))

        nodekey1,   nodekey2    , nodekey3 = self.convo12(nodekey,maskk,  shape_k[1])
        nodevalue1, nodevalue2, nodevalue3 = self.convo12(nodevalue,maskk,shape_k[1])
        nodequery1, nodequery2, nodequery3 = self.convo12(nodequery,maskq,shape_q[1])

        #print('out of conv kkkk',np.shape(nodekey1[0]),np.shape(nodekey2[0]),np.shape(nodekey3[0]))
        #print('out of conv vvvv',np.shape(nodevalue1[0]), np.shape(nodevalue2[0]), np.shape(nodevalue3[0]))
        #print('out of conv qqqq',np.shape(nodequery1[0]), np.shape(nodequery2[0]), np.shape(nodequery3[0]))

        #print(np.shape(nodekey1[0]),np.shape(nodekey2[0]),np.shape(nodekey3[0]))
        #print('********************************************* reaching to attn ')
        #attention
        '''for i in range(0,self.h):
            splited.append(torch.split(torch.transpose(conv1[i],1,2),self.d_k,-1))
        '''

        #print('after all ifff')
        k1, k2, k3 = self.reunion(nodekey1  , nodekey2  , nodekey3)
        q1, q2, q3 = self.reunion(nodequery1, nodequery2, nodequery3)
        v1, v2, v3 = self.reunion(nodevalue1, nodevalue2, nodevalue3)

        #print(np.shape(k1),np.shape(k2),np.shape(k3))
        #print(np.shape(v1))
        #print(np.shape(k))
        #print(np.shape(q1))
        #print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ splited')
        #print(np.shape(splited))
        #print('mask:', np.shape(mask))
        x1, self.attn = attention(q1, k1, v1, mask=mask, dropout=self.dropout)
        x2, self.attn = attention(q2, k2, v2, mask=mask, dropout=self.dropout)
        x3, self.attn = attention(q3, k3, v3, mask=mask, dropout=self.dropout)

        #print('before transpose x',np.shape(x1), np.shape(x2), np.shape(x3))
        #print('self.attn:' ,np.shape(self.attn))
        #print('attention going+++++++++++++++++++++++++++++++++++++++')
        # 3) "Concat" using a view and apply a final linear.
        
        x1 = x1.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x2 = x2.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x3 = x3.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        #print('after transpose x',np.shape(x1), np.shape(x2), np.shape(x3))
        return self.linears[-1](x1+x2+x3)
