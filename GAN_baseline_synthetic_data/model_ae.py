import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, nz, nef):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True), 
            nn.Linear(128, nz))
    def forward(self, input):
        output = self.model(input)
        return output


# class autoencoder(nn.Module):
#     def __init__(self):
#         super(autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(28 * 28, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 64),
#             nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
#         self.decoder = nn.Sequential(
#             nn.Linear(3, 12),
#             nn.ReLU(True),
#             nn.Linear(12, 64),
#             nn.ReLU(True),
#             nn.Linear(64, 128),
#             nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x