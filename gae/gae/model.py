##adapted to pytorch from https://github.com/tkipf/gae and https://github.com/zfjsail/gae-pytorch/blob/master/gae/model.py ##

# from gae.gae.layers import GraphConvolution, InnerProductDecoder
import gae.gae.layers
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNModelAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj)

    def forward(self, x, adj):
        mu= self.encode(x, adj)
        z = mu
        return self.dc(z), mu, None



class GCNModelVAE(nn.Module):
    """
    source: https://github.com/zfjsail/gae-pytorch/blob/master/gae/model.py
    """
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar

class GCNModelVAE3(nn.Module):
    """
    source: https://github.com/zfjsail/gae-pytorch/blob/master/gae/model.py
    """
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2,hidden_dim3, dropout):
        super(GCNModelVAE3, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = gae.gae.layers.GraphConvolution(hidden_dim2, hidden_dim3, dropout, act=lambda x: x)
        self.gc4 = gae.gae.layers.GraphConvolution(hidden_dim2, hidden_dim3, dropout, act=lambda x: x)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        hidden2 = self.gc2(hidden1, adj)
        return self.gc3(hidden2, adj), self.gc4(hidden2, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar
