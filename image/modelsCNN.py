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
        if torch.isnan(torch.sum(h)):
            print('convolution exploded')
        h = h.view(-1, h.size()[1]*h.size()[2]*h.size()[3])
        return self.fcE1(h), self.fcE2(h)
 
    def reparameterize(self, mu, logvar):
#         return mu
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
    
# https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html
class AlexNet(nn.Module):
    def __init__(self, num_classes,regrs=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=11, stride=4, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 96, kernel_size=5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
        self.regrs=regrs
        if regrs:
            self.regrsAct=nn.ReLU()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.regrs:
            x=self.regrsAct(x)
        return x
    
class CNN_VAE_clf(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, fc1,fc2):
        super(CNN_VAE_clf, self).__init__()
 
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
        self.fc=nn.Sequential(
            nn.Linear(fc1, fc2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fc2, fc2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fc2,2)
        )
 
 
    def forward(self, x):
        h=self.encoder(x)
        h = h.view(-1, h.size()[1]*h.size()[2]*h.size()[3])
        return self.fc(h)