import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
import data.dcrnn_utils as dcrnn_utils
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.gnn import gcn, gcn_gwnet, gcn_gcnm_dynamic, spatialGCN
from models.memoryModule import LocalFeatureModule, MemoryModule

class GCNM(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        """

        full_data: full dataset including dateTime
        in_dim: the input data dimension (i.e., node numbers)
        """
        super(GCNM, self).__init__()
        self.local_feature_model = LocalFeatureModule(num_nodes)
        self.memory_model = MemoryModule(in_dim, residual_channels)

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()


        ##s to check if we still need "start_conv"???
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn_gwnet(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field


    def forward(self, input, x_hist):
        """

        :param input: (B, 8, L, D)
        :param x_hist: (B, n*tau, L, D)
        :return: e: enrichied traffic embedding (B, L, D)
        """

        z = self.local_feature_model(input) #(B, L, D)
        z = torch.unsqueeze(z, dim=-1) # (B, L, D) -> (B, L, D, 1)
        x_hist = torch.unsqueeze(x_hist, dim=-1)#(B, n*tau, L, D, 1)
        x_hist = x_hist.transpose(1, 2).contiguous() #(B, L, n*tau, D, F)

        #(B, L, D, F), (B, L, n*tau, D, F)

        e = self.memory_model(z, x_hist) # (B, L, D, F), (B, L, n*tau, D, F) -> (B, F', L, D)

        input = e.permute(0, 1, 3, 2).contiguous()  #(B, F', D, L)

        """
                # the input is from the enriched temporal embedding
                # input: temporal embedding (N, 1, D, L)
                """

        in_len = input.size(3)  # (N, F, D, L), here F=1
        if in_len < self.receptive_field:  # receptive_filed = 12 + 1
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))  # (N, F, D, L+1)
        else:
            x = input
        #x = self.start_conv(x)  # kernel=(1,1), (N, 1, D, L+1) -> (N, 1, D, L+1)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)  # kernel=(1, 2)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)  # kernel=(1,2)
            gate = torch.sigmoid(gate)
            # x=filter=gate: (B, residual_size, D, F)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    # x: (B, residual_size, D, F)
                    #print("input.shape 1 is {}".format(x.size()))
                    x = self.gconv[i](x, new_supports)
                    #print("input.shape 2 is {}".format(x.size()))
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # [N, L, D, 1]
        x = torch.squeeze(x, dim=-1)  # [N, L, D]

        return x.contiguous()

class GCNMdynamic(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        """

        full_data: full dataset including dateTime
        in_dim: the input data dimension (i.e., node numbers)
        """
        super(GCNMdynamic, self).__init__()
        self.local_feature_model = LocalFeatureModule(num_nodes)
        self.memory_model = MemoryModule(in_dim, residual_channels)

        self.num_nodes = num_nodes
        self.device = device
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        ##s to check if we still need "start_conv"???
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 2

        #parameters for initializing the static node embeddings
        node_dim = residual_channels
        self.alpha = 3
        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)
        self.idx = torch.arange(self.num_nodes).to(self.device)

        self.GCN1_1 = gcn_gwnet(c_in=residual_channels,c_out=residual_channels,
                                dropout=self.dropout,support_len=1)

        self.GCN1_2 = gcn_gwnet(c_in=residual_channels,c_out=residual_channels,
                                dropout=self.dropout,support_len=1)

        self.GCN2_1 = gcn_gwnet(c_in=residual_channels,c_out=residual_channels,
                                dropout=self.dropout,support_len=1)

        self.GCN2_2 = gcn_gwnet(c_in=residual_channels,c_out=residual_channels,
                                dropout=self.dropout,support_len=1)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn_gcnm_dynamic(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
        if out_dim > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, out_dim), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, out_dim-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

    def preprocessing(self, adj):
        #adj: (B, L, D, D)
        adj = adj + torch.eye(self.num_nodes).to(self.device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return adj

    def forward(self, input, x_hist):
        """

        :param input: (B, 8, L, D)
        :param x_hist: (B, n*tau, L, D)
        :return: e: enrichied traffic embedding (B, L, D)
        """

        z = self.local_feature_model(input) #(B, L, D)
        z = torch.unsqueeze(z, dim=-1) # (B, L, D) -> (B, L, D, 1)
        x_hist = torch.unsqueeze(x_hist, dim=-1)#(B, n*tau, L, D, 1)
        x_hist = x_hist.transpose(1, 2).contiguous() #(B, L, n*tau, D, F)

        #(B, L, D, F), (B, L, n*tau, D, F)

        e = self.memory_model(z, x_hist) # (B, L, D, F), (B, L, n*tau, D, F) -> (B, residual_channels, L, D)

        input = e.permute(0, 1, 3, 2).contiguous()  #(B, F', D, L)

        """
                # the input is from the enriched temporal embedding
                # input: temporal embedding (N, 1, D, L)
                """

        in_len = input.size(3)  # (N, F, D, L), here F=1
        if in_len < self.receptive_field:  # receptive_filed = 12 + 1
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))  # (N, residual_channels, D, L+1)
        else:
            x = input
        #x = self.start_conv(x)  # kernel=(1,1), (N, 1, D, L+1) -> (N, 1, D, L+1)
        #skip = 0
        skip = self.skip0(x)
        # calculate the current adaptive adj matrix once per iteration
        '''new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]
        '''

        # x: (N, residual_channels, D, L), support[i]: (D, D)
        nodevecInit_1 = self.emb1(self.idx)  # (D, node_dim=residual_channels)
        nodevecInit_2 = self.emb2(self.idx)  # (D, node_dim=residual_channels)
        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)  # kernel=(1, 2)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)  # kernel=(1,2)
            gate = torch.sigmoid(gate)
            # x=filter=gate: (B, residual_size, D, F)
            x = filter * gate

            # ***************** construct dynamic graphs from e ***************** #
            # print("x.size: {}, support0: {}, support1: {}".format(x.size(), self.supports[0].size(), self.supports[1].size()))
            '''filter1 = self.GCN1_1(x, [self.supports[0]]) + self.GCN1_2(x, [
                self.supports[1]])  # (N, residual_channels, D, L)
            filter2 = self.GCN2_1(x, [self.supports[0]]) + self.GCN2_2(x, [
                self.supports[1]])  # (N, residual_channels, D, L)'''
            filter1 = self.GCN1_1(x, [self.supports[0]]) # (N, residual_channels, D, L)
            filter2 = self.GCN2_1(x, [self.supports[1]]) # (N, residual_channels, D, L)
            filter1 = filter1.permute((0, 3, 2, 1)).contiguous()  # (N, L, D, residual_channels)
            filter2 = filter2.permute((0, 3, 2, 1)).contiguous()  # (N, L, D, residual_channels)
            nodevec1 = torch.tanh(self.alpha * torch.mul(nodevecInit_1, filter1))  # (N, L, D, residual_channels)
            nodevec2 = torch.tanh(self.alpha * torch.mul(nodevecInit_2, filter2))

            # objective: construct "support/A" with size (B, D, D, L)
            a = torch.matmul(nodevec1, nodevec2.transpose(2, 3)) - torch.matmul(
                nodevec2, nodevec1.transpose(2, 3))  # (B, L, D, D)
            adj = F.relu(torch.tanh(self.alpha * a))
            mask = torch.zeros(adj.size(0), adj.size(1), adj.size(2), adj.size(3)).to(self.device)
            mask.fill_(float('0'))
            s1, t1 = adj.topk(20, -1)
            mask.scatter_(-1, t1, s1.fill_(1))
            adj = adj * mask

            adp = self.preprocessing(adj)
            adpT = self.preprocessing(adj.transpose(2, 3))
            adp = adp.permute((0, 2, 3, 1)).contiguous()  # (B, D, D, L)
            adpT = adpT.permute((0, 2, 3, 1)).contiguous()
            #new_supports = [adp, adpT, self.supports[0], self.supports[1]]  # dynamic and pre-defined graph
            new_supports = [adp, adpT]

            # parametrized skip connection
            #x = F.dropout(x, self.dropout)
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    # x: (B, residual_size, D, F)
                    #print("input.shape 1 is {}".format(x.size()))
                    x = self.gconv[i](x, new_supports)
                    #print("input.shape 2 is {}".format(x.size()))
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        skip = self.skipE(x) + skip

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x)) # [N, skip_channels, D, 1] -> [N, end_channels, D, 1]
        x = self.end_conv_2(x)  # [N, end_channels, D, 1] -> [N, L, D, 1]
        x = torch.squeeze(x, dim=-1)  # [N, L, D]

        return x.contiguous()

