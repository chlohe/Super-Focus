import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )

class PerceptualAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # nn.Conv2d(128, 32, kernel_size=1, stride=1, padding='same'),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU()
            # nn.MaxPool2d(2),
            # nn.Conv2d(256, 256,kernel_size=3, padding='same'),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU()            
        )
        self.decoder = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(32, 128, kernel_size=1, stride=1, padding='same'),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=6, stride=4, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# model = PerceptualAE()
# x = torch.zeros((1,1,400, 400))
# print(model.forward(x).shape)

class PerceptualAEOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = double_conv(256, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)
        self.dconv_last = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(), 
            nn.Conv2d(64, 1, kernel_size=3, padding='same'),
        )

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_up3(x)
        x = self.upsample(x)

        x = self.dconv_up2(x)        
        x = self.upsample(x)

        x = self.dconv_up1(x)
        x = self.upsample(x)
        
        x = self.dconv_last(x)
        out = torch.sigmoid(x)
        return out