import torch
import torch.nn as nn
import torch.nn.parallel
class Generator(nn.Module):
    def __init__(self, nz, ngf):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ngf, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, ndf, np):#np指的是添加噪声的不同方式的总数
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Linear(128, ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ndf, ndf),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf*8) x 4 x 4
        )
        self.linear1 = nn.Sequential(nn.Linear(ndf, 1), nn.Sigmoid())
        self.linear2 = nn.Linear(ndf, np)
    def forward(self, input):
        output = self.main(input)
        output1 = self.linear1(output)# True or False
        output2 = self.linear2(output)
        return output1, output2