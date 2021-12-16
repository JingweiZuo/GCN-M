import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

def retrieve_hist(dateTime, dataset):
    """

    :param dateTime: (N, L)
    :param dataset:
    :return:
    """
class linear2d(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear2d, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class linear3d(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear3d, self).__init__()
        self.mlp = torch.nn.Conv3d(c_in, c_out, kernel_size=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class MemoryModule(nn.Module):
    def __init__(self, c_in, c_out):
        # c_in: local features z, by default: c_in = 1
        # c_out: output channel, by default: c_out=32, same to the "start_conv" in the main model
        super(MemoryModule, self).__init__()
        # embeddings for Query, Input Memory, Output Memory
        self.mlp_q = linear2d(c_in, c_out)
        self.mlp_m = linear3d(c_in, c_out)
        self.mlp_c = linear3d(c_in, c_out)

    def forward(self, x_local, x_hist):
        """
            ccompute the weighted sum from historical samples
        :param x_local: (B, L, D, F), (batch_size, L, nbr_node, nbr_feature), nbr_feature=1 or 3
        :param x_hist: (B, L, n*tau, D, F), (B, L, n*tau, nbr_node, nbr_feature), nbr_feature=1 or 3
        :return: e: (B, L, 1, D) (batch_size, L, nbr_feature, nbr_node), enriched traffic embeddings
        """

        # x_local: (B, L, D, F) -> (B, F, L, D)
        x_local = x_local.permute((0, 3, 1, 2)).contiguous()
        # x_hist: (B, L, n*tau, D, F) -> (B, F, n*tau, L, D)
        x_hist = x_hist.permute((0, 4, 2, 1, 3)).contiguous()
        q = self.mlp_q(x_local) #(B, F, L, D) -> (B, F', L, D)
        q = torch.unsqueeze(q, dim=-1) #(B, F', L, D) -> (B, F', L, D, 1)

        m = self.mlp_m(x_hist) #(B, F, n*tau, L, D) -> (B, F', n*tau, L, D)
        c = self.mlp_c(x_hist) #(B, F, n*tau, L, D) -> (B, F', n*tau, L, D)
        m = m.permute((0, 1, 3, 2, 4)).contiguous()  # (B, F', n*tau, L, D) -> (B, F', L, n*tau, D)
        c = c.permute((0, 1, 3, 2, 4)).contiguous()  # (B, F', n*tau, L, D) -> (B, F', L, n*tau, D)

        mq = torch.matmul(m, q) #(B, F', L, n*tau, 1)
        mq = torch.squeeze(mq, dim=-1) #(B, F', L, n*tau)
        q = torch.squeeze(q, dim=-1)  # (B, F', L, D)
        attention_scores = F.softmax(F.relu(mq), dim=-1) ##(B, F', L, n*tau)

        o = torch.einsum('bflt,bfltd->bfld', (attention_scores, c)).contiguous() #(B, F', L, D)
        e = q.add(o)

        return e #(B, F', L, D)


class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        print("self.filter_square_matrix.shape is ", self.filter_square_matrix.size(), "self.weight.shape is ", self.weight.size())

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #         print(self.weight.data)
    #         print(self.bias.data)

    def forward(self, input):
        #         print(self.filter_square_matrix.mul(self.weight))
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')'

class LocalFeatureModule(nn.Module):
    def __init__(self, input_size):
        """
        Learn local statistic features: temporal aspect & spatial aspect

            input_size: input dimension size, i.e., node numbers
            X_mean: the mean of the historical input data -> change to the mean of previous L values
        """

        super(LocalFeatureModule, self).__init__()
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size).cuda()
            self.zeros = Variable(torch.zeros(input_size).cuda())
        else:
            self.identity = torch.eye(input_size)
            self.zeros = Variable(torch.zeros(input_size))


        self.gamma_x_t = FilterLinear(input_size, input_size, self.identity)
        self.gamma_x_s = FilterLinear(input_size, input_size, self.identity)

    def step(self, x, mask, x_last_obsv, x_mean_t, delta_t, x_closest_obsv, x_mean_s, delta_s):
        batch_size = x.shape[0]
        dim_size = x.shape[1]

        gama_t = torch.exp(-torch.max(self.zeros, self.gamma_x_t(delta_t)))
        gama_s = torch.exp(-torch.max(self.zeros, self.gamma_x_s(delta_s)))
        x = mask * x + (1 - mask) * (gama_t * x_last_obsv + (1 - gama_t) * x_mean_t
                                     + gama_s * x_closest_obsv + (1 - gama_s) * x_mean_s)

        return x

    def forward(self, input):
        """
            Extract spatio-temporal statistic features
        :param input: (N, 8, L, D)
        :return: outputs: (N, L, D)
        """
        batch_size = input.size(0)
        type_size = input.size(1)
        step_size = input.size(2)
        spatial_size = input.size(3)

        X = torch.squeeze(input[:, 0, :, :])
        Mask = torch.squeeze(input[:, 1, :, :])
        X_last_obsv = torch.squeeze(input[:, 2, :, :])
        X_mean_t = torch.squeeze(input[:, 3, :, :])
        Delta_t = torch.squeeze(input[:, 4, :, :])
        X_closest_obsv = torch.squeeze(input[:, 5, :, :])
        X_mean_s = torch.squeeze(input[:, 6, :, :])
        Delta_s = torch.squeeze(input[:, 7, :, :])


        z_out = None
        for i in range(step_size):
            #params of self.step(): x, mask, x_last_obsv, x_mean_t, delta_t, x_closest_obsv, x_mean_s, delta_s
            res_x = self.step(torch.squeeze(X[:, i:i + 1, :]) \
                              , torch.squeeze(Mask[:, i:i + 1, :]) \
                              , torch.squeeze(X_last_obsv[:, i:i + 1, :]) \
                              , torch.squeeze(X_mean_t[:, i:i + 1, :]) \
                              , torch.squeeze(Delta_t[:, i:i + 1, :]) \
                              , torch.squeeze(X_closest_obsv[:, i:i + 1, :]) \
                              , torch.squeeze(X_mean_s[:, i:i + 1, :]) \
                              , torch.squeeze(Delta_s[:, i:i + 1, :])
                              )

            if z_out is None:
                z_out = res_x.unsqueeze(1) # (N, 1, D)
            else:
                z_out = torch.cat((z_out, res_x.unsqueeze(1)), 1)
        return z_out # (N, L, D)

