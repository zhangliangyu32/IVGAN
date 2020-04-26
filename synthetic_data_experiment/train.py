from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy
from model import *
from dataset import *
# from simple_ae import Encoder

from IPython import embed

a = 1
b = -1
c = 0

gamma = 0.5
alpha = 0.2
eta = 0.5

nz = 3
np = 3
width = 1

lr = 2e-4
lr_encoder = 0.01
batchSize = 64
nepochs = 20
beta1 = 0.5 # 'beta1 for adam. default=0.5'
weight_decay_coeff = 5e-4 # weight decay coefficient for training netE.
default_device = 'cuda:4'

parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='./trained-models', help='folder to output model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual random seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

dataset = SyntheticData()
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                         shuffle=True)


device = torch.device(default_device)

# custom weights initialization called on netG and netD
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1 and not (m.weight is None):
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
#     elif classname.find('Affine') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)

netE = Encoder().to(device)
print(netE)
# encoder = torch.load('./sim_encoder.pth')


netG = Generator().to(device)
print(netG)

netD = Discriminator().to(device)
print(netD)

criterion_reconstruct = nn.L1Loss()
criterion = nn.CrossEntropyLoss()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=lr_encoder, betas=(beta1, 0.999), weight_decay=weight_decay_coeff)



for epoch in range(nepochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        # netD.zero_grad()
        real = data[0].to(device)
        batch_size = real.size(0)
        noise = torch.randn(batch_size, nz, device=device)
        latent_real = netE(real)
        
        output = netD(real)[0] # unnormalized
        errD = torch.mean((output - a) ** 2)
        D_x = torch.sigmoid(output).mean().item()
        output = netD(netG(noise))[0]
        errD += torch.mean((output - b) ** 2)
        D_Gz = torch.sigmoid(output).mean().item()

        errG = torch.mean((output - c) ** 2) + 0.0

        k = torch.randint(np, (1,), dtype=torch.int64).item()
        noise_mask = torch.zeros((batch_size, nz), device=device)
        real_mask = torch.ones((batch_size, nz), device=device)
        index = torch.tensor(range(k * width, (k + 1) * width), dtype=torch.int64, device=device)
        noise_mask = noise_mask.index_fill_(1, index, 1)
        real_mask = real_mask.index_fill_(1, index, 0)
        latent = torch.mul(latent_real, real_mask) + torch.mul(noise, noise_mask)
        fake = netG(latent)
        label = torch.full((batch_size,), k, device=device, dtype=torch.int64)
        output = netD(fake)[1]
        CE_regularizer = gamma * criterion(output, label)
        errD += CE_regularizer
        errG -= CE_regularizer
        errG += eta * criterion_reconstruct(real, netG(latent_real))
        errG += eta * criterion_reconstruct(latent, netE(fake))
        
        optimizerD.zero_grad()
        errD.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizerG.zero_grad()
        errG.backward()
        optimizerG.step()

        ############################
        # (3) Update E network: minimize reconstruction error
        ###########################

        netE.train()
        for j in range(10):
            GAN_loss = torch.tensor(0.0, device=device)
            noise_mask = torch.zeros((batch_size, nz), device=device)
            real_mask = torch.ones((batch_size, nz), device=device)
            index = torch.tensor(range(k * width, (k + 1) * width), dtype=torch.int64, device=device)
            noise_mask = noise_mask.index_fill_(1, index, 1)
            real_mask = real_mask.index_fill_(1, index, 0)
            latent = torch.mul(netE(real), real_mask) + torch.mul(noise, noise_mask)
            fake = netG(latent)
            label = torch.full((batch_size,), k, device=device, dtype=torch.int64)
            output = netD(fake)[1]
            GAN_loss += gamma * criterion(output, label)
            err_reconstruct = criterion_reconstruct(real, netG(netE(real)))
            err_reconstruct += criterion_reconstruct(latent, netE(fake.detach()))
            errE = err_reconstruct + alpha * GAN_loss
            optimizerE.zero_grad()
            errE.backward()
            optimizerE.step()
        netE.eval()        
        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x):%.4f D(G(z)):%.4f CE_regularizer: %.4f Reconstruct_err: %.4f'
            % (epoch, nepochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_Gz, 0 - gamma * CE_regularizer.item(), err_reconstruct))

torch.save(netG.state_dict(), '%s/final_netG.pth' % (opt.outf))
print("netG saved.")
torch.save(netE.state_dict(), '%s/final_netE.pth' % (opt.outf))
print("netE saved.")