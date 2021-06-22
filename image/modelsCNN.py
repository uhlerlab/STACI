#adapted from https://github.com/uhlerlab/cross-modal-autoencoders
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class CNN_VAE(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, fc1,fc2):
        super(CNN_VAE, self).__init__()
 
        self.nc = nc
        self.hidden5=hidden5
        self.fc1=fc1
        self.fc2=fc2
 
        self.encoder = nn.Sequential(
            # input is nc x imsize x imsize
            nn.Conv2d(nc, hidden1, kernel, stride, padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden1) x imsize/stride^2
            nn.Conv2d(hidden1, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden2, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden3, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden4, hidden5, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden5),
            nn.LeakyReLU(0.2, inplace=True),
        )
 
        self.fcE1 = nn.Linear(fc1, fc2)
        self.fcE2 = nn.Linear(fc1,fc2)
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden5, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden4, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden3, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden2, hidden1, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden1, nc, kernel, stride, padding, bias=False),
            nn.Sigmoid(),
        )
 
        self.fcD1 = nn.Sequential(
            nn.Linear(fc2, fc1),
            nn.ReLU(inplace=True),
            )
#         self.bn_mean = nn.BatchNorm1d(fc2)
 
    def encode(self, x):
        h = self.encoder(x)
#         print(h.size())
        h = h.view(-1, h.size()[1]*h.size()[2]*h.size()[3])
        return self.fcE1(h), self.fcE2(h)
 
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
 
    def decode(self, z):
        h = self.fcD1(z)
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1/self.hidden5)), int(np.sqrt(self.fc1/self.hidden5)))
        return self.decoder(h)
 
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar