import torch
import torch.nn as nn
import torch.nn.parallel
# class CostumeAffine(nn.Module):
#     def __init__(self, input_features):
#         super(CostumeAffine, self).__init__()
#         self.input_features = input_features
#         self.weight = nn.Parameter(torch.Tensor(input_features, 1, 1))
#         self.bias = nn.Parameter(torch.Tensor(input_features, 1, 1))
#     def forward(self, input):
#         return input.mul(self.weight) + self.bias
class Generator(nn.Module):
    def __init__(self, ngpu, nz, img_size, nc):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, nc, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, ngpu, img_size, nc, np):#np指的是添加噪声的不同方式的总数
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(nc, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4 + 1
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)
        self.classifier = nn.Linear(128 * ds_size ** 2, 4)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        which_block = self.classifier(out)

        return validity, which_block
