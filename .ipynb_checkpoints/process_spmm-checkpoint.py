import numpy as np
import torch
import pickle
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from torch_geometric.datasets import Reddit2, Flickr
import scipy.sparse as sp
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor, remove_diag, set_diag, spmm, spspmm, get_diag
from torch_sparse import sum as ts_sum
from torch_sparse import mul as ts_mul
from torch_sparse import transpose as ts_trans
import argparse
from tqdm import tqdm
import time
import json
import math

import pynvml

import sys
import os

data_root = './data/'

parser = argparse.ArgumentParser()
parser.add_argument('--datastr',type=str,default='ogbn-arxiv',help='name of dataset you use')
parser.add_argument('--filter',type=str,
                    choices=['adjacency','exact1hop'],
                    default='adjacency',
                    help='graph filter for convolution')
parser.add_argument('--nhop',type=int,default=5,help='number of hops')
parser.add_argument("--gpu", default=0, type=int, help="which GPU to use")
parser.add_argument("--inductive", type=bool, default=False, help="inductive or transductive setting")

def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')

def minibatch_normalization(adj,N,k=1000000):
    deg = ts_sum(adj, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    deg_inv_col = deg_inv_sqrt.view(N, 1).to(device)
    deg_inv_row = deg_inv_sqrt.view(1, N).to(device)

    adj = adj.coo()
    for i in tqdm(range(len(adj[0])//k+1)): 
        tmp = SparseTensor(row=adj[0][i*k:(i+1)*k], 
                           col=adj[1][i*k:(i+1)*k], 
                              # value=adj[2][i*k:(i+1)*k], 
                           sparse_sizes=(N,N)).to(device)
        tmp = ts_mul(tmp, deg_inv_col)
        tmp = ts_mul(tmp, deg_inv_row).to('cpu').coo()
        if i == 0:
            adj_t = [tmp[0],tmp[1],tmp[2]]
        else:
            for _ in range(3):
                adj_t[_] = torch.concat([adj_t[_],tmp[_]],dim=0)
    adj_t = SparseTensor(row=adj_t[0],
                         col=adj_t[1],
                         value=adj_t[2],
                         sparse_sizes=(N,N))
    del deg_inv_col, deg_inv_row, tmp
    torch.cuda.empty_cache()
    return adj_t

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 else torch.device("cpu")

#Load dataset
if args.datastr[:4] == "ogbn":
    dataset = PygNodePropPredDataset(args.datastr,root=data_root)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    #get split indices
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
else:
    if args.datastr == "reddit":
        dataset = Reddit2(root=data_root+args.datastr+"/")
    elif args.datastr == "flickr":
        dataset = Flickr(root=data_root+args.datastr+"/")
    data = dataset[0]
    train_idx = torch.flatten((dataset[0].train_mask==True).nonzero())
    valid_idx = torch.flatten((dataset[0].val_mask==True).nonzero())
    test_idx = torch.flatten((dataset[0].test_mask==True).nonzero())

t_start = time.time()

edge_index = data.edge_index
num_edge = len(edge_index[0])
N = data.num_nodes
labels = data.y.data

row,col = edge_index
adj = SparseTensor(row=row,col=col,sparse_sizes=(N,N))
adj = adj.to_scipy(layout='csr')
print("Getting undirected matrix...")
adj = adj + adj.transpose()
print("Saving unnormalized adjacency matrix")
del edge_index

if args.inductive:
    adj = adj[train_idx, :][:, train_idx]
    N = len(train_idx)
adj = adj.tocoo()
adj = SparseTensor(row=torch.LongTensor(adj.row), col=torch.LongTensor(adj.col), sparse_sizes=(N,N))

if not os.path.exists("precomputation_data/"+args.datastr):
    os.mkdir("precomputation_data/"+args.datastr)

if not os.path.exists("precomputation_data/"+args.datastr+'/feature_training.pickle'):
    feat = data.x.numpy()
    feat = torch.from_numpy(feat).float()
    with open("precomputation_data/"+args.datastr+'/feature_training.pickle',"wb") as fopen:
        pickle.dump(feat[train_idx,:],fopen)
    with open("precomputation_data/"+args.datastr+'/feature_validation.pickle',"wb") as fopen:
        pickle.dump(feat[valid_idx,:],fopen)
    with open("precomputation_data/"+args.datastr+'/feature_test.pickle',"wb") as fopen:
        pickle.dump(feat[test_idx,:],fopen)
    del feat

if args.datastr == "ogbn-papers100M":
    a = 3
    b = 10
    sep_att = 16
else:
    a=b=sep_att = 1

torch.tensor([0]).to(device)
filt = args.filter
k=adj.nnz()//a+1
print(filt,": Normalizing matrix A...")
if filt == 'adjacency':
    adj_tag = "adj"
    t_tmp = time.time()
    adj = set_diag(adj,1)
    k=adj.nnz()//a+1
    adj_mat = minibatch_normalization(adj,N,k=k)
elif filt == 'exact1hop':
    adj_tag = "adj_i"
    t_tmp = time.time()
    adj_mat = minibatch_normalization(remove_diag(adj,0),N,k=k)
del adj

if args.inductive:
    agg_feat = data.x.numpy()[train_idx]
else:
    agg_feat = data.x.numpy()
agg_feat = torch.from_numpy(agg_feat).float()
d = data.num_features
del data

adj_mat = adj_mat.coo()
#  / 1e9))
tmp = SparseTensor(row=adj_mat[0][0:k],
                              col=adj_mat[1][0:k],
                              value=adj_mat[2][0:k],
                              sparse_sizes=(N,N))
tmp = tmp.coo() 
k_adj = math.ceil(len(adj_mat[0])//b)
k_att = math.ceil(d//sep_att)

torch.cuda.empty_cache()
with torch.no_grad():
    for _ in range(args.nhop): # number of hops
        t_tmp = time.time()
        feat_list=[]
        for i_feat in tqdm(range(sep_att)):
            agg_feat_block = agg_feat[:,i_feat*(k_att):(i_feat+1)*(k_att)]
            agg_feat_block = agg_feat_block.to(device)
            for i_adj in range(b): # matrix separation for saving GPU memory
                tmp = torch.sparse.FloatTensor(
                    torch.stack([adj_mat[0][i_adj*k_adj:(i_adj+1)*k_adj],adj_mat[1][i_adj*k_adj:(i_adj+1)*k_adj]]),
                    adj_mat[2][i_adj*k_adj:(i_adj+1)*k_adj],
                    [N,N]).to(device)
                torch.cuda.empty_cache()
                if i_adj == 0:
                    tmp_feat = torch.spmm(tmp, agg_feat_block)
                else:
                    tmp_feat += torch.spmm(tmp, agg_feat_block)
            feat_list.append(tmp_feat.to('cpu'))
            del tmp_feat, agg_feat_block
            torch.cuda.empty_cache()
        agg_feat = torch.concat(feat_list, dim=1)
        print(str(_+1)+"hop finished")

        if args.inductive:
            with open("precomputation_data/"+args.datastr+'/'+filt+'_'+str(_+1)+'_training.pickle',"wb") as fopen:
                pickle.dump(agg_feat,fopen)
        else:
            with open("precomputation_data/"+args.datastr+'/'+filt+'_'+str(_+1)+'_training.pickle',"wb") as fopen:
                pickle.dump(agg_feat[train_idx,:],fopen)
            with open("precomputation_data/"+args.datastr+'/'+filt+'_'+str(_+1)+'_validation.pickle',"wb") as fopen:
                pickle.dump(agg_feat[valid_idx,:],fopen)
            with open("precomputation_data/"+args.datastr+'/'+filt+'_'+str(_+1)+'_test.pickle',"wb") as fopen:
                pickle.dump(agg_feat[test_idx,:],fopen)
            
del agg_feat, adj_mat, tmp
print(f'GPU max memory usage: {torch.cuda.max_memory_allocated(device=device)/10**9}')

#save labels
if not os.path.exists("precomputation_data/"+args.datastr+'/labels.pickle'):
    labels_train = labels[train_idx].reshape(-1).long()
    labels_valid = labels[valid_idx].reshape(-1).long()
    labels_test = labels[test_idx].reshape(-1).long()
    with open("precomputation_data/"+args.datastr+'/labels.pickle',"wb") as fopen:
        pickle.dump([labels_train,labels_valid,labels_test],fopen)
        