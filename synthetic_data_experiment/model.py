import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, width=10):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(3, width),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(width, width),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(width, 3)
        )
    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):
    def __init__(self, width=10):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(3, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 3),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, width=10):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(3, width),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.linear1 = nn.Linear(width, 1)
        self.linear2 = nn.Linear(width, 3)
    def forward(self, input):
        output = self.main(input)
        return self.linear1(output), self.linear2(output)

