# imports
import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3,
                 red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()

        # branch 1: 1x1
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)

        # branch 2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # branch 3: 1x1 -> 5x5
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        # branch 4: 3x3 pool -> 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)


class GoogLeNET(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogLeNET, self).__init__()

        # initial layers
        self.conv1 = conv_block(in_channels=in_channels, out_channels=64,
                                kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception modules (channels match original GoogLeNet structure)
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)   # output channels: 256
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64) # output channels: 480
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)  # -> 512
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64) # -> 512
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64) # -> 512
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64) # -> 528
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128) # -> 832
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128) # -> 832
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128) # -> 1024

        # final pooling / classifier
        # Use AdaptiveAvgPool2d to be flexible with input size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)                  # shape: (N, 1024, 1, 1)
        x = torch.flatten(x, 1)              # shape: (N, 1024)
        x = self.dropout(x)
        x = self.fc(x)                       # shape: (N, num_classes)
        return x


# Example usage (for testing):
# model = GoogLeNET(in_channels=3, num_classes=1000)
# x = torch.randn(1, 3, 224, 224)
# out = model(x)
# print(out.shape)   # expected: torch.Size([1, 1000])
