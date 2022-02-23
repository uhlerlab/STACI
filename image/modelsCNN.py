import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Printer(nn.Module):
    def __init__(self):
        super(Printer,self).__init__()
    def forward(self,x):
        print(x.shape)
        return x

class CNN_VAE(nn.Module):
#adapted from https://github.com/uhlerlab/cross-modal-autoencoders
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
    
class CNN_VAE_hook(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, fc1,fc2):
        super(CNN_VAE_hook, self).__init__()
 
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
        self.gradients = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def encode(self, x):
        h = self.encoder(x)
        hook = h.register_hook(self.activations_hook)
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
    
    def get_activations_gradient(self):
        return self.gradients
    def get_activations(self, x):
        return self.encoder(x)
    
class CNN_VAE_sharded(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, fc1,fc2):
        super(CNN_VAE_sharded, self).__init__()
 
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
        self.encoder.cuda(0)
 
        self.fcE1 = nn.Linear(fc1, fc2)
        self.fcE2 = nn.Linear(fc1,fc2)
        self.fcE1.cuda(3)
        self.fcE2.cuda(3)
 
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
        self.decoder.cuda(0)
 
        self.fcD1 = nn.Sequential(
            nn.Linear(fc2, fc1),
            nn.ReLU(inplace=True),
            )
        self.fcD1.cuda(0)
#         self.bn_mean = nn.BatchNorm1d(fc2)
 
    def encode(self, x):
        h = self.encoder(x.cuda(0).float())
#         print(h.size())
        if torch.isnan(torch.sum(h)):
            print('convolution exploded')
        h = h.view(-1, h.size()[1]*h.size()[2]*h.size()[3])
        h=h.cuda(3)
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
        h = self.fcD1(z.cuda(0))
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1/self.hidden5)), int(np.sqrt(self.fc1/self.hidden5)))
        return self.decoder(h)
 
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar

class CNN_VAE_alexnet(nn.Module):
    def __init__(self, num_classes):
        super(CNN_VAE_alexnet, self).__init__()
 
        self.num_classes=num_classes
 
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=12, stride=4, padding=2),
#             Printer(),
            nn.LeakyReLU(inplace=True),
#             Printer(),
            nn.Conv2d(32, 96, kernel_size=6,stride=2, padding=2),
#             Printer(),
            nn.LeakyReLU(inplace=True),
#             Printer(),
            nn.Conv2d(96, 192, kernel_size=4,stride=2, padding=1),
#             Printer(),
            nn.LeakyReLU(inplace=True),
#             Printer(),
            nn.Conv2d(192, 128, kernel_size=4,stride=2, padding=1),
#             Printer(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=4,stride=2, padding=1),
#             Printer(),
            nn.LeakyReLU(inplace=True),
#             Printer(),
        )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.encoder_linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(18432, 6000),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(6000, 6000),
            nn.LeakyReLU(inplace=True),
#             nn.Linear(2048, num_classes),
        )
 
        self.fcE1 = nn.Linear(6000, num_classes)
        self.fcE2 = nn.Linear(6000, num_classes)
 

        self.decoder_linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_classes, 6000),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(6000, 6000),
            nn.LeakyReLU(inplace=True),
            nn.Linear(6000, 18432),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4,stride=2, padding=1),
#             Printer(),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 192, kernel_size=4,stride=2, padding=1),
#             Printer(),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, kernel_size=4,stride=2, padding=1),
#             Printer(),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(96, 32, kernel_size=6,stride=2, padding=2),
#             Printer(),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=12, stride=4, padding=2),
#             Printer(),
            nn.Sigmoid(),
        )

#         self.fcD1 = nn.Sequential(
#             nn.Linear(fc2, fc1),
#             nn.ReLU(inplace=True),
#             )
#         self.bn_mean = nn.BatchNorm1d(fc2)
 
    def encode(self, x):
        x = self.encoder(x)
#         print(x.shape)
#         x = self.avgpool(x)
        x = torch.flatten(x, 1)
#         print(x.shape)
        x = self.encoder_linear(x)
        return self.fcE1(x), self.fcE2(x)
 
    def reparameterize(self, mu, logvar):
#         return mu
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
 
    def decode(self, z):
        h = self.decoder_linear(z)
        h = h.view(-1, 128, 12, 12)
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
    
class FC_l3(nn.Module):
    def __init__(self,inputdim,fcdim1,fcdim2,fcdim3, num_classes,dropout=0.5,regrs=True):
        super(FC_l3, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(inputdim, fcdim1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fcdim1, fcdim2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fcdim2, fcdim3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fcdim3, num_classes),            
        )
        self.regrs=regrs
        if regrs:
            self.regrsAct=nn.ReLU()

    def forward(self, x):
        x = self.classifier(x)
        if self.regrs:
            x=self.regrsAct(x)
        return x   

class FC_l1(nn.Module):
    def __init__(self,inputdim,fcdim1, num_classes,regrs=True):
        super(FC_l1, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(inputdim, fcdim1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fcdim1, num_classes),            
        )
        self.regrs=regrs
        if regrs:
            self.regrsAct=nn.ReLU()

    def forward(self, x):
        x = self.classifier(x)
        if self.regrs:
            x=self.regrsAct(x)
        return x   
    
class FC_l0(nn.Module):
    def __init__(self,inputdim, num_classes,regrs=True):
        super(FC_l0, self).__init__()
        
        self.classifier = nn.Linear(inputdim, num_classes)         
        self.regrs=regrs
        if regrs:
            self.regrsAct=nn.ReLU()

    def forward(self, x):
        x = self.classifier(x)
        if self.regrs:
            x=self.regrsAct(x)
        return x 
    
class FC_l5(nn.Module):
    def __init__(self,inputdim,fcdim1,fcdim2,fcdim3,fcdim4,fcdim5, num_classes,dropout=0.5,regrs=True):
        super(FC_l5, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(inputdim, fcdim1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fcdim1, fcdim2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fcdim2, fcdim3),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fcdim3, fcdim4),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fcdim4, fcdim5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fcdim5, num_classes),            
        )
        self.regrs=regrs
        if regrs:
            self.regrsAct=nn.ReLU()

    def forward(self, x):
        x = self.classifier(x)
        if self.regrs:
            x=self.regrsAct(x)
        return x   