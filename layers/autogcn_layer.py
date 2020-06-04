import torch
import torch.nn as nn
import numpy as np
from dgl.nn.pytorch import APPNPConv
import dgl.function as fn
from torch.nn import init


class MPConv(nn.Module):
    def __init__(self):
        super(MPConv, self).__init__()

    def forward(self, graph, feat):
        graph = graph.local_var()
        norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = torch.reshape(norm, shp).to(feat.device)

        # normalization by src node
        feat = feat * norm
        graph.ndata['h'] = feat
        graph.update_all(fn.copy_src(src='h', out='m'),
                         fn.sum(msg='m', out='h'))

        feat = graph.ndata['h']
        # normalization by dst node
        feat = feat * norm
        return feat


class AUTOGCNLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """

    def __init__(self, in_dim, out_dim, activation, dropout, graph_norm, batch_norm, K=8, num_high=1, num_low=1, num_mid=1, residual=False, gate=True, opt='over'):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.gate = gate
        if in_dim != out_dim:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.num_low = num_low
        self.num_high = num_high
        self.num_mid = num_mid
        
        self.low = nn.ModuleList()
        self.high = nn.ModuleList()
        self.mid = nn.ModuleList()
        self.low_gamma = nn.ParameterList()
        self.mid_gamma = nn.ParameterList()
        self.high_gamma = nn.ParameterList()

        for i in range(self.num_low):
            self.low.append(nn.Linear(in_dim, out_dim,bias=False))

        for i in range(self.num_high):
            self.high.append(nn.Linear(in_dim, out_dim,bias=False))

        for i in range(self.num_mid):
            self.mid.append(nn.Linear(in_dim, out_dim,bias=False))

        self.eps = 1e-9
        self.K = K
        self.mp = MPConv()
        self.opt = opt

        if self.opt== 'over':
            for i in range(self.num_low):
                self.low_gamma.append(torch.nn.Parameter(torch.FloatTensor([1/K for i in range(K)])))
            for i in range(self.num_mid):
                self.mid_gamma.append(torch.nn.Parameter(torch.FloatTensor([1/K for i in range(K)])))
            for i in range(self.num_high):
                self.high_gamma.append(torch.nn.Parameter(torch.FloatTensor([1/K for i in range(K)])))
            self.alpha = torch.Tensor(np.linspace(self.eps, 1-self.eps, self.K))
            self.midalpha = torch.Tensor(np.linspace(self.eps, 1, self.K))

        elif self.opt== 'single':
            self.lowalpha = torch.nn.Parameter(torch.FloatTensor([0 for i in range(self.num_low)]))
            self.lowgamma = torch.nn.Parameter(torch.FloatTensor([1 for i in range(self.num_low)]))
            self.highalpha = torch.nn.Parameter(torch.FloatTensor([0 for i in range(self.num_high)]))
            self.highgamma = torch.nn.Parameter(torch.FloatTensor([1 for i in range(self.num_high)]))
            self.midalpha = torch.nn.Parameter(torch.FloatTensor([0 for i in range(self.num_mid)]))
            self.midgamma = torch.nn.Parameter(torch.FloatTensor([1 for i in range(self.num_mid)]))

        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            init.zeros_(self.bias)


    def forward(self, g, feature, snorm_n):
        h = self.mp(g, feature)
        out = []

        if self.opt== 'over':
            alpha = self.alpha.to(feature.device)
            for i in range(self.num_low):
                gamma = self.low_gamma[i]
                gamma = torch.relu(gamma)
                gamma = gamma.squeeze()
                a = torch.dot(alpha, gamma)
                b = torch.dot(1 - alpha, gamma)
                o = a * h + b * feature
                o = self.low[i](o)
                out.append(o)
            for i in range(self.num_high):
                gamma = self.high_gamma[i]
                gamma = torch.relu(gamma)
                gamma = gamma.squeeze()
                a = torch.dot(-alpha, gamma)
                b = torch.dot(1 - alpha, gamma)
                o = a * h + b * feature
                o = self.high[i](o)
                out.append(o)
            midalpha = self.midalpha.to(feature.device)
            h1 = self.mp(g, h)
            for i in range(self.num_mid):
                gamma = self.mid_gamma[i]
                gamma = torch.relu(gamma)
                gamma = gamma.squeeze()
                a = torch.sum(gamma)
                c = torch.dot(midalpha,gamma)
                o = a*h1-c*feature
                o = self.mid[i](o)
                out.append(o)

        elif self.opt == 'single':
            for i in range(self.num_low):
                lowalpha = torch.sigmoid(self.lowalpha)
                lowgamma = torch.relu(self.lowgamma)
                highalpha = torch.sigmoid(self.highalpha)
                highgamma = torch.relu(self.highgamma)
                midalpha = torch.sigmoid(self.midalpha)
                midgamma = torch.relu(self.midgamma)

                for i in range(self.num_low):
                    o = (lowalpha[i] * h + (1 - lowalpha[i]) * feature) * lowgamma[i]
                    o = self.low[i](o)
                    out.append(o)
                for i in range(self.num_high):
                    o = (-highalpha[i] * h + (1 - highalpha[i]) * feature) * highgamma[i]
                    o = self.high[i](o)
                    out.append(o)
                h1 = self.mp(g, h)
                for i in range(self.num_mid):
                    o = (h1 - midalpha[i] * feature) * midgamma[i]
                    o = self.mid[i](o)
                    out.append(o)

                self.lowalpha_value = [t.item() for t in lowalpha]
                self.lowgamma_value = [t.item() for t in lowgamma]
                self.highalpha_value = [t.item() for t in highalpha]
                self.highgamma_value = [t.item() for t in highgamma]
                self.midalpha_value = [t.item() for t in midalpha]
                self.midgamma_value = [t.item() for t in midgamma]

        elif self.opt == 'fix':
            for i in range(self.num_low):
                o = 0.5 * h + 0.5 * feature
                o = self.low[i](o)
                out.append(o)
            for i in range(self.num_high):
                o = -0.5 * h + 0.5 * feature
                o = self.high[i](o)
                out.append(o)
            h1 = self.mp(g, h)
            for i in range(self.num_mid):
                o = (h1 - 0.5 * feature)
                o = self.mid[i](o)
                out.append(o)

        if self.gate and self.num_low==1 and self.num_mid==1 and self.num_high==1:
            out[0] = out[0] * (torch.sigmoid(out[1] + out[2]))
            out[1] = out[1] * (torch.sigmoid(out[0] + out[2]))
            out[2] = out[2] * (torch.sigmoid(out[0] + out[1]))

        if self.gate and self.num_low==0 or self.num_mid==0 or self.num_high==0:
            out[0] = out[0] * (torch.sigmoid(out[1]))
            out[1] = out[1] * (torch.sigmoid(out[0]))

        out = [o.unsqueeze(0) for o in out]
        out = torch.cat(out, dim=0)
        out = torch.sum(out, dim=0)
        out = out.squeeze()
        out = out + self.bias

        if self.graph_norm:
            out = out * snorm_n  # normalize activation w.r.t. graph size

        if self.batch_norm:
            out = self.batchnorm_h(out)  # batch normalization

        if self.activation:
            out = self.activation(out)

        if self.residual:
            out = feature + out  # residual connection
        out = self.dropout(out)
        return out

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                                                         self.in_channels,
                                                                         self.out_channels, self.residual)
