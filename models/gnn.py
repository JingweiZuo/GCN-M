import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()

class nconv_gcnm_dynamic(nn.Module):
    def __init__(self):
        super(nconv_gcnm_dynamic, self).__init__()

    def forward(self, x, A):
        #x: (B, F, D, L), A: (B, D, D, L)
        #return (B, F, D, L)
        x = torch.einsum('ncvl,nvwl->ncwl', (x, A))
        return x.contiguous()

class nconv_gwnet(nn.Module):
    def __init__(self):
        super(nconv_gwnet, self).__init__()

    def forward(self, x, A):
        #print("x.size is {}, A.size is {}".format(x.size(), A.size()))
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class nconv_spatialGCN(nn.Module):
    def __init__(self):
        super(nconv_spatialGCN,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ndl, dw-> nwl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            #print("x.size is {}, a.size is {}".format(x.size(), a.size()))
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training) # (N, C, W, L)
        return h

class gcn_gcnm_dynamic(nn.Module):
    def __init__(self,c_in=32,c_out=32,dropout=0.3,support_len=3,order=2):
        super(gcn_gcnm_dynamic,self).__init__()
        self.nconv_dynamic = nconv_gcnm_dynamic()
        self.nconv_static = nconv_gwnet()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        # x: (B, F, D, L), support/A: (B, D, D, L) or (D, D)
        out = [x] # x.size(): (N, D, L)
        for a in support[:2]: #Dynamic Adjacency Matrix
            x1 = self.nconv_dynamic(x,a) #a: (B, D, D, L)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv_dynamic(x1,a)
                out.append(x2)
                x1 = x2
        if len(support) > 2:
            for a in support[2:]:  # Pre-defined static adjacency matrix
                x1 = self.nconv_static(x, a)  # a: (D, D)
                out.append(x1)
                for k in range(2, self.order + 1):
                    x2 = self.nconv_static(x1, a)
                    out.append(x2)
                    x1 = x2
        h = torch.cat(out,dim=1)  #(N, c_in, D, L)
        h = self.mlp(h) #[N, c_out, D, L]
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gcn_gwnet(nn.Module):
    def __init__(self,c_in=32,c_out=32,dropout=0.3,support_len=3,order=2):
        super(gcn_gwnet,self).__init__()
        self.nconv = nconv_gwnet()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x] # x.size(): (N, c_in, D, L)
        for a in support: #2 direction adjacency matrix
            x1 = self.nconv(x,a) #x1: (N, c_in, D, L)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)  #(N, c_in, D, L)
        h = self.mlp(h) #[N, c_out, D, L]
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class linear_spatialGCN(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear_spatialGCN,self).__init__()
        self.mlp = torch.nn.Conv1d(c_in, c_out, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn_spatialGCN(nn.Module):
    def __init__(self,c_in=32,c_out=32,dropout=0.3,support_len=3,order=2):
        super(gcn_spatialGCN,self).__init__()
        self.nconv = nconv_spatialGCN()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x] # x.size(): (N, D, L)
        for a in support: #2 direction adjacency matrix
            x1 = self.nconv(x,a) #x1: (N, D, L)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)  #(N, D*nbr, L)
        h = self.mlp(h) #[N, D, L]
        h = F.dropout(h, self.dropout, training=self.training)
        return h

# consider the adaptive learnable graph
class spatialGCN(nn.Module): #inspired by GraphWaveNet
    def __init__(self, device, num_nodes, c_g_in=207, c_out=512, dropout=0.3, supports=None, layers=2):
        super(spatialGCN, self).__init__()
        # use several gcn layers
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.supports = supports
        self.layers = layers
        self.mlp = linear_spatialGCN(c_g_in, c_out)

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if supports is None:
            self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len += 1

        for i in range(layers):
            self.gconv.append(gcn(c_g_in, c_g_in, dropout, support_len=self.supports_len))
            self.bn.append(nn.BatchNorm1d(c_g_in)) #(N, D, L) 3D

    def forward(self, x):
        # calculate the current adaptive adj matrix once per iteration
        x = x.permute(0, 2, 1) #input: (N, L, D)
        x = x.contiguous() #out: (N, D, L)
        new_supports = None
        if self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1) #N x N
            new_supports = self.supports + [adp]
        for i in range(self.layers):
            if self.supports is not None:
                x = self.gconv[i](x, new_supports)  #x:[N, D, L], out: #[N, D, L]
                x = self.bn[i](x) #out: (N, D, L)
        x =self.mlp(x) #out: (N, D', L)
        x = x.permute(0, 2, 1)
        x = x.contiguous() #out: (N, L, D')
        return x
