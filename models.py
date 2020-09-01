import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    
    def __init__(self):
        
        super(Generator, self).__init__()
        
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.channels,self.img_rows,self.img_cols)
        self.latent_dim = 128
        
        net = []
        
        for i in range(1,4):
            
            net.append(nn.Linear(self.latent_dim*2**(i-1), self.latent_dim*2**i))
            net.append(nn.LeakyReLU(0.2, inplace=True))
            net.append(nn.BatchNorm1d(self.latent_dim*2**i,0.8))
            
        net.append(nn.Linear(1024, np.prod(self.img_shape)))
        net.append(nn.Tanh())
        
        self.model = nn.Sequential(*net)


    def forward(self, noise):
        img = self.model(noise)
        return img.view(img.size(0), *self.img_shape)




class Discriminator(nn.Module):
    
    def __init__(self):
        
        super(Discriminator, self).__init__()
        
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.channels,self.img_rows,self.img_cols)
        
        self.model = nn.Sequential(
            nn.Linear(np.prod(self.img_shape), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    def forward(self, img):
        img = img.view(img.size(0), -1)
        return self.model(img)



