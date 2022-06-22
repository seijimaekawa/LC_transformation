import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
    
class GCN_LC(nn.Module):
    def __init__(self,nfeat,nclass,nhidden,dropout):
        super(GCN_LC, self).__init__()
        self.W1 = nn.Linear(nfeat, nhidden)
        self.W2 = nn.Linear(nhidden, nclass)
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.reset_parameters()
        
    def reset_parameters(self):
        self.W1.reset_parameters()
        self.W2.reset_parameters()

    def forward(self,list_adj,layer_norm,device,st=0,end=0):
        x = list_adj[0][st:end,:].to(device)
        x = self.W1(x)
        x = self.act_fn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.W2(x)
        return F.log_softmax(x,dim=1)
    
class JKnet_LC(nn.Module):
    def __init__(self,nfeat,nlayers,nclass,nhidden,dropout,pooling="max"):
        super(JKnet_LC, self).__init__()
        self.W1 = nn.Linear(nfeat, nhidden)
        # self.W2 = nn.Linear(nhidden, nhidden)
        self.W = nn.ModuleList([nn.Linear(nhidden,nhidden) for _ in range(nlayers)])
        if pooling == "concat":
            self.W_final = nn.Linear(nhidden*nlayers, nclass)
        self.pooling = pooling
        self.nlayers = nlayers
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.reset_parameters()
        
    def reset_parameters(self):
        self.W1.reset_parameters()
        for ind in range(len(self.W)):
            self.W[ind].reset_parameters()
        if self.pooling == "concat":
            self.W_final.reset_parameters()
            
    def forward(self,list_adj,layer_norm,device,st=0,end=0):
        batch_size = end-st
        x_list = []
        for ind, mat in enumerate(list_adj[1:]):
             x_list.append(mat[st:end,:].to(device))
        x = torch.concat(x_list,dim=0)
        x = self.W1(x)
        x_list = [x[:batch_size]]
        for i in range(1,self.nlayers):
            x = x[batch_size:]
            x = self.act_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.W[i](x)
            x_list.append(x[:batch_size])
        if self.pooling == "concat":
            x = torch.concat(x_list,dim=1)
            x = self.act_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.W_final(x)
        elif self.pooling == "max":
            x = torch.stack(x_list).max(dim=0).values
        return F.log_softmax(x,dim=1)

class GPRGNN_LC3(nn.Module):
    def __init__(self, nfeat,nlayers,nhidden,nclass,dropout,dp,alpha):
        super(GPRGNN_LC3, self).__init__()
        self.lin1 = nn.Linear(nfeat, nhidden)
        self.lin2 = nn.Linear(nhidden, nhidden)
        self.lin3 = nn.Linear(nhidden, nclass)

        TEMP = alpha*(1-alpha)**np.arange(nlayers+1)
        TEMP[-1] = (1-alpha)**nlayers
        self.temp = nn.Parameter(torch.tensor(TEMP, dtype=torch.float32))

        self.dropout = dropout
        self.dprate = dp
        self.reset_parameters

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self,list_adj,layer_norm,device,st=0,end=0):
        x = []
        for ind, mat in enumerate(list_adj):
            x.append(mat[st:end,:].to(device))
        x = torch.concat(x, dim=0)
            
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        x = F.dropout(x, p=self.dprate, training=self.training)
        
        nlayers = len(list_adj)
        batch_size = end-st
        for i in range(nlayers):
            ind_st = batch_size*i
            ind_end = batch_size*(i+1)
            if i == 0:
                gamma = self.temp[i]
                hidden = x[ind_st:ind_end]*gamma
            else:
                gamma = self.temp[i]
                hidden = hidden + x[ind_st:ind_end]*gamma
        return F.log_softmax(hidden, dim=1)
    
class GPRGNN_LC4(nn.Module):
    def __init__(self, nfeat,nlayers,nhidden,nclass,dropout,dp,alpha):
        super(GPRGNN_LC4, self).__init__()
        self.lin1 = nn.Linear(nfeat, nhidden)
        self.lin2 = nn.Linear(nhidden, nhidden)
        self.lin3 = nn.Linear(nhidden, nhidden//2)
        self.lin4 = nn.Linear(nhidden//2, nclass)

        TEMP = alpha*(1-alpha)**np.arange(nlayers+1)
        TEMP[-1] = (1-alpha)**nlayers
        self.temp = nn.Parameter(torch.tensor(TEMP, dtype=torch.float32))

        self.dropout = dropout
        self.dprate = dp
        self.reset_parameters

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()

    def forward(self,list_adj,layer_norm,device,st=0,end=0):
        x = []
        for ind, mat in enumerate(list_adj):
            x.append(mat[st:end,:].to(device))
        x = torch.concat(x, dim=0)
            
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin3(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin4(x)
        x = F.dropout(x, p=self.dprate, training=self.training)
        
        nlayers = len(list_adj)
        batch_size = end-st
        for i in range(nlayers):
            ind_st = batch_size*i
            ind_end = batch_size*(i+1)
            if i == 0:
                gamma = self.temp[i]
                hidden = x[ind_st:ind_end]*gamma
            else:
                gamma = self.temp[i]
                hidden = hidden + x[ind_st:ind_end]*gamma
        return F.log_softmax(hidden, dim=1)

# from https://github.com/Tiiiger/SGC
class SGC(nn.Module):
    def __init__(self,nfeat,nclass):
        super(SGC, self).__init__()
        self.W = nn.Linear(nfeat, nclass)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self,list_adj,layer_norm,device,st=0,end=0):
        for ind, mat in enumerate(list_adj):
            mat = mat[st:end,:].to(device)
        x = self.W(mat)
        return F.log_softmax(x,dim=1)   
    
# from https://github.com/sunilkmaurya/FSGNN
class FSGNN_Large(nn.Module):
    def __init__(self,nfeat,nlayers,nhidden,nclass,dp1,dp2):
        super(FSGNN_Large,self).__init__()
        self.wt1 = nn.ModuleList([nn.Linear(nfeat,int(nhidden)) for _ in range(nlayers)])
        self.fc2 = nn.Linear(nhidden*nlayers,nhidden)
        self.fc3 = nn.Linear(nhidden,nclass)
        self.dropout1 = dp1 
        self.dropout2 = dp2 
        self.act_fn = nn.ReLU()
        self.att = nn.Parameter(torch.ones(nlayers))
        self.sm = nn.Softmax(dim=0)
        self.reset_parameters()

    def reset_parameters(self):
        for ind in range(len(self.wt1)):
            self.wt1[ind].reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        
    def forward(self,list_adj,layer_norm,device,st=0,end=0):
        mask = self.sm(self.att)
        mask = torch.mul(len(list_adj),mask)
        list_out = list()
        for ind, mat in enumerate(list_adj):
            mat = mat[st:end,:].to(device)
            tmp_out = self.wt1[ind](mat)
            if layer_norm == True:
                tmp_out = F.normalize(tmp_out,p=2,dim=1)
            tmp_out = torch.mul(mask[ind],tmp_out)
            list_out.append(tmp_out)
        final_mat = torch.cat(list_out, dim=1)
        out = self.act_fn(final_mat)
        out = F.dropout(out,self.dropout1,training=self.training)
        out = self.fc2(out)
        out = self.act_fn(out)
        out = F.dropout(out,self.dropout2,training=self.training)
        out = self.fc3(out)
        return F.log_softmax(out, dim=1)    
    
if __name__ == '__main__':
    pass






