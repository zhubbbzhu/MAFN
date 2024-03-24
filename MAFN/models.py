import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sympy.tensor import tensor
from torch.nn import Linear

from layers import GraphConvolution
import torch
import sympy

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x




class decoder(torch.nn.Module):
    def __init__(self, nfeat,  nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 64)

    def forward(self, mlp_in):
        weight_output =self.wl(mlp_in)

        return weight_output

# 定义xinmlp模型
class xinmlp(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(xinmlp, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 64)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class MAFN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(MAFN, self).__init__()
        
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.ZINB = decoder(nfeat, nhid1, nhid2)
        self.dropout = dropout
        self.meta = nn.Parameter(torch.Tensor([0.1]))
        self.meta.data.clamp_(0, 1)

        self.MLP = nn.Sequential(
            nn.Linear(192, 64)
        )
        self.MLP_L=MLP_L(64)


    def forward(self, x, sadj, fadj):
        emb1 = self.SGCN(x, sadj)  # Spatial_GCN

        emb2 = self.FGCN(x, fadj)  # Feature_GCN
        conadj = self.meta * fadj + (1 - self.meta) * sadj
        com = self.CGCN(x, conadj)

        emb = torch.stack([emb1, com, emb2], dim=1)
        a = self.MLP_L(emb)
        emb = F.normalize(a, p=2)

        emb = torch.cat((emb[:, 0].mul(emb1), emb[:, 1].mul(com), emb[:, 2].mul(emb2)), 1)
        emb = self.MLP(emb)

        [pi, disp, mean] = self.ZINB(emb)
        return emb, pi, disp, mean, emb1, emb2, com

