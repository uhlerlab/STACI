##adapted to pytorch from https://github.com/tkipf/gae and https://github.com/zfjsail/gae-pytorch/blob/master/gae/model.py ##

# from gae.gae.layers import GraphConvolution, InnerProductDecoder
import gae.gae.layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2,hidden_dim3, dropout):
        super(GCNModelVAE3, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
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

class GCNModelVAE_XA(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2,fc_dim1,fc_dim2,fc_dim3,fc_dim4, dropout):
        super(GCNModelVAE_XA, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fc1 = gae.gae.layers.FC(hidden_dim2,fc_dim1, dropout)
        self.fc2 = gae.gae.layers.FC(fc_dim1, fc_dim2, dropout)
        self.fc3 = gae.gae.layers.FC(fc_dim2, fc_dim3, dropout)
        self.fc4 = gae.gae.layers.FC(fc_dim3, fc_dim4, dropout)
        self.fc5 = gae.gae.layers.FC(fc_dim4, input_feat_dim, dropout, act = lambda x: x, batchnorm = False)

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

    def decode_X(self,z):
        output = self.fc1(z)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        output = self.fc5(output)
        return output
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z, self.decode_X(z)

class GCNModelVAE_XA_e3(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2,hidden_dim3,fc_dim1,fc_dim2,fc_dim3,fc_dim4, dropout):
        super(GCNModelVAE_XA_e3, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.gc3 = gae.gae.layers.GraphConvolution(hidden_dim2, hidden_dim3, dropout, act=lambda x: x)
        self.gc4 = gae.gae.layers.GraphConvolution(hidden_dim2, hidden_dim3, dropout, act=lambda x: x)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fc1 = gae.gae.layers.FC(hidden_dim3,fc_dim1, dropout)
        self.fc2 = gae.gae.layers.FC(fc_dim1, fc_dim2, dropout)
        self.fc3 = gae.gae.layers.FC(fc_dim2, fc_dim3, dropout)
        self.fc4 = gae.gae.layers.FC(fc_dim3, fc_dim4, dropout)
        self.fc5 = gae.gae.layers.FC(fc_dim4, input_feat_dim, dropout, act = lambda x: x, batchnorm = False)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        hidden2 = self.gc2(hidden1,adj)
        return self.gc3(hidden2, adj), self.gc4(hidden2, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        output = self.fc1(z)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        output = self.fc5(output)
        return output
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z, self.decode_X(z)
    
class GCNModelVAE_XA_e1(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, dropout):
        super(GCNModelVAE_XA_e1, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.leaky_relu)
        self.gc1s = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.leaky_relu)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fc1 = gae.gae.layers.FC(hidden_dim1, input_feat_dim, dropout, act = lambda x: x, batchnorm = True)

    def encode(self, x, adj):
        return self.gc1(x, adj), self.gc1s(x, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        output = self.fc1(z)
        return output
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z, self.decode_X(z)
    
class GCNModelVAE_XA_e2_d1(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1,hidden_dim2, dropout):
        super(GCNModelVAE_XA_e2_d1, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.leaky_relu)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.gc2s = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fc1 = gae.gae.layers.FC(hidden_dim2, input_feat_dim, dropout, act = lambda x: x, batchnorm = True)

    def encode(self, x, adj):
        hidden1=self.gc1(x,adj)
        return self.gc2(hidden1, adj), self.gc2s(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        output = self.fc1(z)
        return output
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z, self.decode_X(z)
    
class GCNModelVAE_XA_e2_d1_DCA(nn.Module):   
    def __init__(self, input_feat_dim, hidden_dim1,hidden_dim2,hidden_decoder, dropout,meanMin=1e-5,meanMax=1e6,thetaMin=1e-5,thetaMax=1e6):
        super(GCNModelVAE_XA_e2_d1_DCA, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.leaky_relu)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.gc2s = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fc1 = gae.gae.layers.FC(hidden_dim2, hidden_decoder, dropout, act = F.leaky_relu, batchnorm = True)
        self.pi=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = torch.sigmoid, batchnorm = False,bias=True)
        self.theta=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(F.softplus(x),min=thetaMin,max=thetaMax), batchnorm = False,bias=True)
        self.mean=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(torch.exp(x),min=meanMin,max=meanMax), batchnorm = False,bias=True)

    def encode(self, x, adj):
        hidden1=self.gc1(x,adj)
        return self.gc2(hidden1, adj), self.gc2s(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        output = self.fc1(z)
        pi_res=self.pi(output)
        theta_res=self.theta(output)
        mean_res=self.mean(output)
        return output,pi_res,theta_res,mean_res
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z, self.decode_X(z)

class GCNModelVAE_XA_e2_d1_DCA_sharded(nn.Module):   
    def __init__(self, input_feat_dim, hidden_dim1,hidden_dim2,hidden_decoder, dropout,meanMin=1e-5,meanMax=1e6,thetaMin=1e-5,thetaMax=1e6):
        super(GCNModelVAE_XA_e2_d1_DCA_sharded, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.leaky_relu)
        self.gc1.cuda(0)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.gc2.cuda(1)
        self.gc2s = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.gc2s.cuda(2)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.dc.cuda(0)
        self.fc1 = gae.gae.layers.FC(hidden_dim2, hidden_decoder, dropout, act = F.leaky_relu, batchnorm = True)
        self.fc1.cuda(3)
        self.pi=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = torch.sigmoid, batchnorm = False,bias=True)
        self.pi.cuda(2)
        self.theta=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(F.softplus(x),min=thetaMin,max=thetaMax), batchnorm = False,bias=True)
        self.theta.cuda(3)
        self.mean=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(torch.exp(x),min=meanMin,max=meanMax), batchnorm = False,bias=True)
        self.mean.cuda(3)

    def encode(self, x, adj):
        x=x.cuda(0).float()
        adj=adj.cuda(0).float()
        hidden1=self.gc1(x,adj)
#         hidden1=hidden1.cuda(1)
#         adj=adj.cuda(1)
        return self.gc2(hidden1.cuda(1), adj.cuda(1)), self.gc2s(hidden1.cuda(2), adj.cuda(2))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        z=z.cuda(3)
        output = self.fc1(z)
#         output=output.cuda(3)
        pi_res=self.pi(output.cuda(2))
        theta_res=self.theta(output)
        mean_res=self.mean(output)
        return output.cuda(0),pi_res.cuda(0),theta_res.cuda(0),mean_res.cuda(0)
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar.cuda(1))
        return self.dc(z.cuda(0)), mu, logvar, z, self.decode_X(z)

class GCNModelVAE_XA_e2_d1_DCA_fca(nn.Module):   
    def __init__(self, input_feat_dim, hidden_dim1,hidden_dim2,hidden_decoder, dropout,meanMin=1e-5,meanMax=1e6,thetaMin=1e-5,thetaMax=1e6):
        super(GCNModelVAE_XA_e2_d1_DCA_fca, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.leaky_relu)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.gc2s = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fc1 = gae.gae.layers.FC(hidden_dim2, hidden_decoder, dropout, act = F.leaky_relu, batchnorm = True)
        self.fca = gae.gae.layers.FC(hidden_dim2, hidden_dim2, dropout, act = F.leaky_relu, batchnorm = True)
        self.pi=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = torch.sigmoid, batchnorm = False,bias=True)
        self.theta=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(F.softplus(x),min=thetaMin,max=thetaMax), batchnorm = False,bias=True)
        self.mean=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(torch.exp(x),min=meanMin,max=meanMax), batchnorm = False,bias=True)

    def encode(self, x, adj):
        hidden1=self.gc1(x,adj)
        return self.gc2(hidden1, adj), self.gc2s(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        output = self.fc1(z)
        pi_res=self.pi(output)
        theta_res=self.theta(output)
        mean_res=self.mean(output)
        return output,pi_res,theta_res,mean_res
    
    def decode_A(self,z):
        output = self.fca(z)
        return self.dc(output)
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.decode_A(z), mu, logvar, z, self.decode_X(z)
    
class GCNModelVAE_XA_e2_d1_DCAfork(nn.Module):   
    def __init__(self, input_feat_dim, hidden_dim1,hidden_dim2,hidden_decoder, dropout,meanMin=1e-5,meanMax=1e6,thetaMin=1e-5,thetaMax=1e6):
        super(GCNModelVAE_XA_e2_d1_DCAfork, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.leaky_relu)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.gc2s = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fc1p = gae.gae.layers.FC(hidden_dim2, hidden_decoder, dropout, act = F.leaky_relu, batchnorm = True)
        self.fc1t = gae.gae.layers.FC(hidden_dim2, hidden_decoder, dropout, act = F.leaky_relu, batchnorm = True)
        self.fc1m = gae.gae.layers.FC(hidden_dim2, hidden_decoder, dropout, act = F.leaky_relu, batchnorm = True)
        self.pi=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = torch.sigmoid, batchnorm = False,bias=True)
        self.theta=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(F.softplus(x),min=thetaMin,max=thetaMax), batchnorm = False,bias=True)
        self.mean=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(torch.exp(x),min=meanMin,max=meanMax), batchnorm = False,bias=True)

    def encode(self, x, adj):
        hidden1=self.gc1(x,adj)
        return self.gc2(hidden1, adj), self.gc2s(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        outputp = self.fc1p(z)
        outputt = self.fc1t(z)
        outputm = self.fc1m(z)
        pi_res=self.pi(outputp)
        theta_res=self.theta(outputt)
        mean_res=self.mean(outputm)
        return (outputp,outputt,outputm),pi_res,theta_res,mean_res
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z, self.decode_X(z)

class GCNModelVAE_XA_e2_d1_DCAelemPi(nn.Module):   
    def __init__(self, input_feat_dim, hidden_dim1,hidden_dim2,hidden_decoder, dropout,shareGenePi,meanMin=1e-5,meanMax=1e6,thetaMin=1e-5,thetaMax=1e6):
        super(GCNModelVAE_XA_e2_d1_DCAelemPi, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.leaky_relu)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.gc2s = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fc1 = gae.gae.layers.FC(hidden_dim2, hidden_decoder, dropout, act = F.leaky_relu, batchnorm = True)
        if shareGenePi:
            pisize=1
        else:
            pisize=input_feat_dim
        self.pi=gae.gae.layers.FC_elementwise(input_feat_dim,pisize, dropout=0, act = torch.sigmoid, batchnorm = False,bias=True)
        self.theta=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(F.softplus(x),min=thetaMin,max=thetaMax), batchnorm = False,bias=True)
        self.meanNoAct=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: x, batchnorm = False,bias=True)
        self.meanAct=lambda x: torch.clamp(torch.exp(x),min=meanMin,max=meanMax)

    def encode(self, x, adj):
        hidden1=self.gc1(x,adj)
        return self.gc2(hidden1, adj), self.gc2s(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        output = self.fc1(z)
#         pi_res=self.pi(output)
        theta_res=self.theta(output)
        mean=self.meanNoAct(output)
        mean_res=self.meanAct(mean)
        pi_res=self.pi(mean)
        return output,pi_res,theta_res,mean_res
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z, self.decode_X(z)

class GCNModelVAE_XA_e2_d1_DCA_constantDisp(nn.Module):   
    def __init__(self, input_feat_dim, hidden_dim1,hidden_dim2,hidden_decoder, dropout,meanMin=1e-5,meanMax=1e6,thetaMin=1e-5,thetaMax=1e6):
        super(GCNModelVAE_XA_e2_d1_DCA_constantDisp, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.leaky_relu)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.gc2s = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fc1 = gae.gae.layers.FC(hidden_dim2, hidden_decoder, dropout, act = F.leaky_relu, batchnorm = True)
        self.pi=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = torch.sigmoid, batchnorm = False,bias=True)
        self.theta=nn.parameter.Parameter(torch.zeros(1,input_feat_dim))
        self.thetaAct=lambda x: torch.clamp(F.softplus(x),min=thetaMin,max=thetaMax)
        self.mean=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(torch.exp(x),min=meanMin,max=meanMax), batchnorm = False,bias=True)

    def encode(self, x, adj):
        hidden1=self.gc1(x,adj)
        return self.gc2(hidden1, adj), self.gc2s(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        output = self.fc1(z)
        pi_res=self.pi(output)
        theta_res=self.thetaAct(self.theta)
        mean_res=self.mean(output)
        return output,pi_res,theta_res,mean_res
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z, self.decode_X(z)    

class GCNModelVAE_XA_e2_d1_DCAshared(nn.Module):   
    def __init__(self, input_feat_dim, hidden_dim1,hidden_dim2,hidden_decoder, dropout,meanMin=1e-5,meanMax=1e6,thetaMin=1e-5,thetaMax=1e6):
        super(GCNModelVAE_XA_e2_d1_DCAshared, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.leaky_relu)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.gc2s = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fc1 = gae.gae.layers.FC(hidden_dim2, hidden_decoder, dropout, act = F.leaky_relu, batchnorm = True)
        self.pi=gae.gae.layers.FC(hidden_decoder, 1, dropout=0, act = torch.sigmoid, batchnorm = False,bias=True)
        self.theta=gae.gae.layers.FC(hidden_decoder, 1, dropout=0, act = lambda x: torch.clamp(F.softplus(x),min=thetaMin,max=thetaMax), batchnorm = False,bias=True)
        self.mean=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(torch.exp(x),min=meanMin,max=meanMax), batchnorm = False,bias=True)

    def encode(self, x, adj):
        hidden1=self.gc1(x,adj)
        return self.gc2(hidden1, adj), self.gc2s(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        output = self.fc1(z)
        pi_res=self.pi(output)
        theta_res=self.theta(output)
        mean_res=self.mean(output)
        return output,pi_res,theta_res,mean_res
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z, self.decode_X(z)
    
class GCNModelVAE_XA_e4_d1(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1,hidden_dim2,hidden_dim3,hidden_dim4, dropout):
        super(GCNModelVAE_XA_e4_d1, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.leaky_relu)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.leaky_relu)
        self.gc3 = gae.gae.layers.GraphConvolution(hidden_dim2, hidden_dim3, dropout, act=F.leaky_relu)
        self.gc4 = gae.gae.layers.GraphConvolution(hidden_dim3, hidden_dim4, dropout, act=F.leaky_relu)
        self.gc4s = gae.gae.layers.GraphConvolution(hidden_dim3, hidden_dim4, dropout, act=F.leaky_relu)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fc1 = gae.gae.layers.FC(hidden_dim1, input_feat_dim, dropout, act = lambda x: x, batchnorm = True)

    def encode(self, x, adj):
        hidden=self.gc1(x,adj)
        hidden=self.gc2(hidden,adj)
        hidden=self.gc3(hidden,adj)
        return self.gc4(hidden, adj), self.gc4s(hidden, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        output = self.fc1(z)
        return output
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z, self.decode_X(z)    
    
class GCNModelVAE_gcnX_inprA(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2,gcn_dim1, dropout):
        super(GCNModelVAE_gcnX_inprA, self).__init__()
        self.gc1 = gae.gae.layers.GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = gae.gae.layers.GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.gc_dec1 = gae.gae.layers.GraphConvolution(hidden_dim2,gcn_dim1, dropout,act=F.relu)
        self.gc_dec2 = gae.gae.layers.GraphConvolution(gcn_dim1, input_feat_dim, dropout,act=lambda x: x)

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

    def decode_X(self,z,adj_decode):
        output = self.gc_dec1(z,adj_decode)
        output = self.gc_dec2(output,adj_decode)
        return output
    
    def forward(self, x, adj,adj_decode=None):
        if adj_decode is None:
            adj_decode=adj
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z, self.decode_X(z,adj_decode)
    

class FCVAE(nn.Module): 
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2,hidden_dim3,hidden_dim4,hidden_dim5,fc_dim1,fc_dim2,fc_dim3,fc_dim4, dropout):
        super(FCVAE, self).__init__()
        self.fcE1 = gae.gae.layers.FC(input_feat_dim,hidden_dim1, dropout)
        self.fcE2 = gae.gae.layers.FC(hidden_dim1, hidden_dim2, dropout)
        self.fcE3 = gae.gae.layers.FC(hidden_dim2, hidden_dim3, dropout)
        self.fcE4 = gae.gae.layers.FC(hidden_dim3, hidden_dim4, dropout)
        self.fcE5 = gae.gae.layers.FC(hidden_dim4, hidden_dim5, dropout, act=lambda x: x, batchnorm = False)
        self.fcE5s = gae.gae.layers.FC(hidden_dim4, hidden_dim5, dropout, act=lambda x: x, batchnorm = False)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fcD1 = gae.gae.layers.FC(hidden_dim5,fc_dim1, dropout)
        self.fcD2 = gae.gae.layers.FC(fc_dim1, fc_dim2, dropout)
        self.fcD3 = gae.gae.layers.FC(fc_dim2, fc_dim3, dropout)
        self.fcD4 = gae.gae.layers.FC(fc_dim3, fc_dim4, dropout)
        self.fcD5 = gae.gae.layers.FC(fc_dim4, input_feat_dim, dropout, act = lambda x: x, batchnorm = False)

    def encode(self, x):
        hidden = self.fcE1(x)
        hidden = self.fcE2(hidden)
        hidden = self.fcE3(hidden)
        hidden = self.fcE4(hidden)
        return self.fcE5(hidden),self.fcE5s(hidden)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        output = self.fcD1(z)
        output = self.fcD2(output)
        output = self.fcD3(output)
        output = self.fcD4(output)
        output = self.fcD5(output)
        return output
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z, self.decode_X(z)

class FCVAE1(nn.Module): 
    def __init__(self, input_feat_dim, hidden, dropout):
        super(FCVAE1, self).__init__()
        self.fcE1 = gae.gae.layers.FC(input_feat_dim, hidden, dropout, act=F.leaky_relu, batchnorm = True)
        self.fcE1s = gae.gae.layers.FC(input_feat_dim, hidden, dropout, act=F.leaky_relu, batchnorm = True)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fcD1 = gae.gae.layers.FC(hidden, input_feat_dim, dropout, act = lambda x:x, batchnorm = True)

    def encode(self, x):
        return self.fcE1(x),self.fcE1s(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        output = self.fcD1(z)
        return output
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z, self.decode_X(z)

class FCVAE1_fca(nn.Module): 
    def __init__(self, input_feat_dim, hidden, dropout):
        super(FCVAE1_fca, self).__init__()
        self.fcE1 = gae.gae.layers.FC(input_feat_dim, hidden, dropout, act=F.leaky_relu, batchnorm = True)
        self.fcE1s = gae.gae.layers.FC(input_feat_dim, hidden, dropout, act=F.leaky_relu, batchnorm = True)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fcD1 = gae.gae.layers.FC(hidden, input_feat_dim, dropout, act = lambda x:x, batchnorm = True)
        self.fcD2 = gae.gae.layers.FC(hidden, hidden, dropout, act = F.leaky_relu, batchnorm = True)

    def encode(self, x):
        return self.fcE1(x),self.fcE1s(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        output = self.fcD1(z)
        return output
    
    def decode_A(self,z):
        output=self.fcD2(z)
        return self.dc(output)
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode_A(z), mu, logvar, z, self.decode_X(z)    
    
class FCVAE1_DCA(nn.Module): 
    def __init__(self, input_feat_dim, hidden,hidden_decoder, dropout,meanMin=1e-5,meanMax=1e6,thetaMin=1e-5,thetaMax=1e6):
        super(FCVAE1_DCA, self).__init__()
        self.fcE1 = gae.gae.layers.FC(input_feat_dim, hidden, dropout, act=F.leaky_relu, batchnorm = True,bias=True)
        self.fcE1s = gae.gae.layers.FC(input_feat_dim, hidden, dropout, act=F.leaky_relu, batchnorm = True,bias=True)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fcD1 = gae.gae.layers.FC(hidden, hidden_decoder, dropout, act = F.leaky_relu, batchnorm = True,bias=True)
        self.pi=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = torch.sigmoid, batchnorm = False,bias=True)
        self.theta=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(F.softplus(x),min=thetaMin,max=thetaMax), batchnorm = False,bias=True)
        self.mean=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(torch.exp(x),min=meanMin,max=meanMax), batchnorm = False,bias=True)
        

    def encode(self, x):
        return self.fcE1(x),self.fcE1s(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        output = self.fcD1(z)
        pi_res=self.pi(output)
        theta_res=self.theta(output)
        mean_res=self.mean(output)
        return output,pi_res,theta_res,mean_res
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z, self.decode_X(z)    

class FCVAE1_DCA_sharded(nn.Module): 
    def __init__(self, input_feat_dim, hidden,hidden_decoder, dropout,meanMin=1e-5,meanMax=1e6,thetaMin=1e-5,thetaMax=1e6):
        super(FCVAE1_DCA_sharded, self).__init__()
        self.fcE1 = gae.gae.layers.FC(input_feat_dim, hidden, dropout, act=F.leaky_relu, batchnorm = True,bias=True)
        self.fcE1.cuda(0)
        self.fcE1s = gae.gae.layers.FC(input_feat_dim, hidden, dropout, act=F.leaky_relu, batchnorm = True,bias=True)
        self.fcE1s.cuda(0)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.dc.cuda(0)
        self.fcD1 = gae.gae.layers.FC(hidden, hidden_decoder, dropout, act = F.leaky_relu, batchnorm = True,bias=True)
        self.fcD1.cuda(2)
        self.pi=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = torch.sigmoid, batchnorm = False,bias=True)
        self.pi.cuda(1)
        self.theta=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(F.softplus(x),min=thetaMin,max=thetaMax), batchnorm = False,bias=True)
        self.theta.cuda(1)
        self.mean=gae.gae.layers.FC(hidden_decoder, input_feat_dim, dropout=0, act = lambda x: torch.clamp(torch.exp(x),min=meanMin,max=meanMax), batchnorm = False,bias=True)
        self.mean.cuda(1)
        

    def encode(self, x):
        x=x.cuda(0)
        return self.fcE1(x),self.fcE1s(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_X(self,z):
        output = self.fcD1(z.cuda(2))
        output=output.cuda(1)
        pi_res=self.pi(output)
        theta_res=self.theta(output)
        mean_res=self.mean(output)
        return output.cuda(0),pi_res.cuda(0),theta_res.cuda(0),mean_res.cuda(0)
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z, self.decode_X(z)  
    
class FCAE(nn.Module): 
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2,hidden_dim3,hidden_dim4,hidden_dim5,fc_dim1,fc_dim2,fc_dim3,fc_dim4, dropout):
        super(FCAE, self).__init__()
        self.fcE1 = gae.gae.layers.FC(input_feat_dim,hidden_dim1, dropout)
        self.fcE2 = gae.gae.layers.FC(hidden_dim1, hidden_dim2, dropout)
        self.fcE3 = gae.gae.layers.FC(hidden_dim2, hidden_dim3, dropout)
        self.fcE4 = gae.gae.layers.FC(hidden_dim3, hidden_dim4, dropout)
        self.fcE5 = gae.gae.layers.FC(hidden_dim4, hidden_dim5, dropout, act=lambda x: x, batchnorm = False)
        self.dc = gae.gae.layers.InnerProductDecoder(dropout, act=lambda x: x)
        self.fcD1 = gae.gae.layers.FC(hidden_dim5,fc_dim1, dropout)
        self.fcD2 = gae.gae.layers.FC(fc_dim1, fc_dim2, dropout)
        self.fcD3 = gae.gae.layers.FC(fc_dim2, fc_dim3, dropout)
        self.fcD4 = gae.gae.layers.FC(fc_dim3, fc_dim4, dropout)
        self.fcD5 = gae.gae.layers.FC(fc_dim4, input_feat_dim, dropout, act = lambda x: x, batchnorm = False)

    def encode(self, x):
        hidden = self.fcE1(x)
        hidden = self.fcE2(hidden)
        hidden = self.fcE3(hidden)
        hidden = self.fcE4(hidden)
        return self.fcE5(hidden)

    def decode_X(self,z):
        output = self.fcD1(z)
        output = self.fcD2(output)
        output = self.fcD3(output)
        output = self.fcD4(output)
        output = self.fcD5(output)
        return output
    
    def forward(self, x, adj):
        mu = self.encode(x)
        z = mu
        return self.dc(z), mu, None, z, self.decode_X(z)
 
class FCAE1(nn.Module): 
    def __init__(self, input_feat_dim, dropout,hidden):
        super(FCAE1, self).__init__()
        self.fcE1 = gae.gae.layers.FC(input_feat_dim, hidden, dropout, act=F.leaky_relu, batchnorm = True)
        
        self.fcD1 = gae.gae.layers.FC(hidden, input_feat_dim, dropout, act = lambda x:x, batchnorm = True)

    def encode(self, x):
        hidden = self.fcE1(x)
        return hidden

    def decode_X(self,z):
        output = self.fcD1(z)
        return output
    
    def forward(self, x, adj):
        mu = self.encode(x)
        z = mu
        return None, mu, None, z, self.decode_X(z)
    
class FCAE2(nn.Module): 
    def __init__(self, input_feat_dim, dropout,hidden1,hidden2):
        super(FCAE2, self).__init__()
        self.fcE1 = gae.gae.layers.FC(input_feat_dim, hidden1, dropout, act=F.leaky_relu, batchnorm = True)
        self.fcE2 = gae.gae.layers.FC(hidden1, hidden2, dropout, act=F.leaky_relu, batchnorm = True)
        self.fcD1 = gae.gae.layers.FC(hidden2, hidden1, dropout, act = F.leaky_relu, batchnorm = True)
        self.fcD2 = gae.gae.layers.FC(hidden1, input_feat_dim, dropout, act = lambda x:x, batchnorm = True)

    def encode(self, x):
        hidden = self.fcE1(x)
        hidden = self.fcE2(hidden)
        return hidden

    def decode_X(self,z):
        output = self.fcD1(z)
        output = self.fcD2(z)
        return output
    
    def forward(self, x, adj):
        mu = self.encode(x)
        z = mu
        return None, mu, None, z, self.decode_X(z)
    
class Clf_fc1(nn.Module): 
    def __init__(self, input_feat_dim, dropout,hidden,out_dim,batchnorm=True,bias=True,act=F.leaky_relu):
        super(Clf_fc1, self).__init__()
        self.fc1 = gae.gae.layers.FC(input_feat_dim, hidden, dropout, act=act, batchnorm = batchnorm,bias=bias)
        
        self.fc2 = gae.gae.layers.FC(hidden, out_dim, dropout, act = lambda x:x, batchnorm = batchnorm,bias=bias)

    
    def forward(self, z):
        out=self.fc1(z)
        return self.fc2(out)
    
class Clf_linear1(nn.Module): 
    def __init__(self, input_feat_dim, dropout,out_dim,batchnorm=True,bias=True):
        super(Clf_linear1, self).__init__()
        
        self.fc1 = gae.gae.layers.FC(input_feat_dim, out_dim, dropout, act = lambda x:x, batchnorm = batchnorm,bias=bias)

    
    def forward(self, z):
        out=self.fc1(z)
        return out