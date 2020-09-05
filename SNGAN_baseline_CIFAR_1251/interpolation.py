from __future__ import print_function
import argparse
import math
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
from model_GAN import *
from model_ae import *
from fid import *
# from simple_ae import Encoder

from IPython import embed

nz = 100 # size of latent variable
ngf = 64 
ndf = 64 
itfr_sigma = {0: 0}
n_critic = 5

lr_D = 1e-4
lr_G = 2e-4
batchSize = 64
imageSize = 32 # 'the height / width of the input image to network'
workers = 2 # 'number of data loading workers'
nepochs = 600
beta1 = 0.5 # 'beta1 for adam. default=0.5'
weight_decay_coeff = 5e-4 # weight decay coefficient for training netE.
default_device = 'cuda:3'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="cifar10", help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', default='/data1/zhangliangyu/datasets/data_cifar10', help='path to dataset')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./trained_model', help='folder to output model checkpoints')
parser.add_argument('--outp', default='./fake-imgs', help='folder to output images')
parser.add_argument('--manualSeed', type=int, help='manual random seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

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

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                                transform=transforms.Compose([
                                transforms.Resize(imageSize),
                                transforms.CenterCrop(imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    nc=3
elif opt.dataset == 'lsun':
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(imageSize),
                            transforms.CenterCrop(imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
    m_true, s_true = compute_dataset_statistics(target_set="LSUN", batch_size=50, dims=2048, cuda=True, device=default_device)
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                            transform=transforms.Compose([
                            transforms.Resize(imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    nc=3
    m_true, s_true = compute_dataset_statistics(target_set="CIFAR10", batch_size=50, dims=2048, cuda=True, device=default_device)

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                            transform=transforms.Compose([
                            transforms.Resize(imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                            ]))
        nc=1

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, imageSize, imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                        shuffle=True, num_workers=int(workers))


ngpu = 1
device = torch.device(default_device)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and not (m.weight is None):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Affine') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def generate_sample(generator, latent_size, num_image=50000, batch_size=50): #generate data sample to compute the fid.
    generator.eval()
    
    z_try = Variable(torch.randn(1, latent_size, 1, 1).to(device))
    data_try = generator(z_try)

    data_sample = numpy.empty((num_image, data_try.shape[1], data_try.shape[2], data_try.shape[3]))
    for i in range(0, num_image, batch_size):
        start = i
        end = i + batch_size
        z = Variable(torch.randn(batch_size, latent_size, 1, 1).to(device))
        d = generator(z)
        data_sample[start:end] = d.cpu().data.numpy()
    
    return data_sample

netG = Generator(ngpu, nz, ngf, nc).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = Discriminator(nc, ndf).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# encoder = torch.load('./sim_encoder.pth')
# encoder.eval()


criterion_BCE = nn.BCELoss()


fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.9))

with open('./fid_record.txt', 'a') as f:
    f.write("fid_record:" + '\n')


for epoch in range(nepochs):
    if epoch in itfr_sigma:
        sigma = itfr_sigma[epoch]
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        # netD.zero_grad()
        real = data[0].to(device)
        batch_size = real.size(0)
        noise = torch.randn(batch_size, nz, 1, 1, device=device)

        pixd_noise = torch.randn(real.size(), device=device)
        pixg_noise = torch.randn(real.size(), device=device)
        
        label_real = torch.full((batch_size,), 1, device=device, dtype=torch.float32)
        output = netD(real + sigma * pixd_noise) # unnormalized
        errD = criterion_BCE(output, label_real)
        D_x = torch.sigmoid(output).mean().item()
        label_fake = torch.full((batch_size,), 0, device=device, dtype=torch.float32)
        output = netD(netG(noise) + sigma * pixg_noise)
        errD += criterion_BCE(output, label_fake)
        D_Gz = torch.sigmoid(output).mean().item()

        
        optimizerD.zero_grad()
        errD.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if i % n_critic == 0:
            errG = criterion_BCE(output, label_real)
            optimizerG.zero_grad()
            errG.backward()
            optimizerG.step()

        ############################
        # (3) Update E network: minimize reconstruction error
        ###########################
        
        if i % 1000 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x):%.4f D(G(z)):%.4f'
            % (epoch, nepochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_Gz))
    
    if (epoch + 1) % 10 == 0:
        vutils.save_image(real, '%s/real_samples.png' % opt.outp, normalize=True)
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (opt.outp, epoch + 1), normalize=True)

        dataset_fake = generate_sample(generator = netG, latent_size = nz)
        fid = calculate_fid(dataset_fake, m_true, s_true, device=default_device)
        print("The Frechet Inception Distance:", fid)
        # do checkpointing
        with open('./fid_record.txt', 'a') as f:
            f.write(str(fid) + '\n')
    

torch.save(netG.state_dict(), '%s/final_netG.pth' % (opt.outf))
torch.save(netD.state_dict(), '%s/final_netD.pth' % (opt.outf))
