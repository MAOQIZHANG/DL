import torch
import torch.nn as nn

class UNet(nn.Module):
    # this UNet model is for image segmentation task
    def __init__(self, in_channels=3, out_channels=49):
        super(UNet, self).__init__()

        # Downsample path convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)

        # Upsample path convolutional layers
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Output convolutional layer, map to out_channels
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Downsample path
        x1 = self.conv1(x)
        x2 = self.conv2(nn.functional.max_pool2d(x1, 2))
        x3 = self.conv3(nn.functional.max_pool2d(x2, 2))
        x4 = self.conv4(nn.functional.max_pool2d(x3, 2))
        x5 = self.conv5(nn.functional.max_pool2d(x4, 2))

        # Upsample path
        x6 = self.upconv1(x5)
        x6 = torch.cat([x4, x6], dim=1)
        x6 = self.conv6(x6)
        x7 = self.upconv2(x6)
        x7 = torch.cat([x3, x7], dim=1)
        x7 = self.conv7(x7)
        x8 = self.upconv3(x7)
        x8 = torch.cat([x2, x8], dim=1)
        x8 = self.conv8(x8)
        x9 = self.upconv4(x8)
        x9 = torch.cat([x1, x9], dim=1)
        x9 = self.conv9(x9)

        # Output
        x10 = self.outconv(x9)
        return x10
