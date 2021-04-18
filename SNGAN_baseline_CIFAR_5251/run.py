import os
import torch
import torch.optim as optim
import torch_mimicry as mmc
from torch_mimicry.nets import sngan
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy
from fid import * 

def generate_sample(generator, latent_size, num_image=10000, batch_size=50): #generate data sample to compute the fid.
    generator.eval()
    
    z_try = Variable(torch.randn(1, latent_size).to(device))
    data_try = generator(z_try)

    data_sample = numpy.empty((num_image, data_try.shape[1], data_try.shape[2], data_try.shape[3]))
    for i in range(0, num_image, batch_size):
        start = i
        end = i + batch_size
        z = Variable(torch.randn(batch_size, latent_size).to(device))
        d = generator(z)
        data_sample[start:end] = d.cpu().data.numpy()
    
    return data_sample

device = torch.device("cuda:4")
num_checkpoints = 40
dataset = dset.CIFAR10(root='/data1/zhangliangyu/datasets/data_cifar10', download=True,
                            transform=transforms.Compose([
                            transforms.Resize(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=True, num_workers=2)

netG = sngan.SNGANGenerator64().to(device)
netD = sngan.SNGANDiscriminator64().to(device)
optD = optim.Adam(netD.parameters(), 5e-5, betas=(0.5, 0.9))
optG = optim.Adam(netG.parameters(), 5e-5, betas=(0.5, 0.9))


m_true, s_true = compute_dataset_statistics(target_set="CIFAR10", batch_size=50, dims=2048, cuda=True, device=device)
fixed_noise = torch.randn(64, 128, device=device)
with open('./fid_record.txt', 'a') as f:
    f.write("fid_record:" + '\n')

for k in range(num_checkpoints):
    trainer = mmc.training.Trainer(
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        n_dis=2,
        num_steps=1000,
        lr_decay=None,
        dataloader=dataloader,
        log_dir="./log",
        device=device)
    trainer.train()

    fake = netG(fixed_noise)
    vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % ('./fake-imgs', k * 10 + 1), normalize=True)

    dataset_fake = generate_sample(generator = netG, latent_size = 128)
    fid = calculate_fid(dataset_fake, m_true, s_true, device=device)
    print("The Frechet Inception Distance:", fid)
    # do checkpointing
    with open('./fid_record.txt', 'a') as f:
        f.write(str(fid) + '\n')
    os.system('clear')
    os.system("rm -r log")