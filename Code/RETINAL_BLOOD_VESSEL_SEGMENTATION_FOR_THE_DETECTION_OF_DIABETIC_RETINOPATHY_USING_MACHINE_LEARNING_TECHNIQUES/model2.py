import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = ConvBlock(in_channels, in_channels)
        self.attention_map = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x) * self.attention_map(x)

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))

class USNet(nn.Module):
    def __init__(self):
        super(USNet, self).__init__()
        self.conv1 = nn.Sequential(ConvBlock(3, 32), ConvBlock(32, 64))
        self.att1 = nn.Sequential(AttentionBlock(64), AttentionBlock(64))
        self.dilatedconv1 = nn.Sequential(DilatedConvBlock(64, 128, 2), DilatedConvBlock(128, 128, 2))
        self.deconv1 = nn.Sequential(DeconvBlock(128, 64), DeconvBlock(64, 32))
        self.conv2 = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.att1(x)
        x = self.dilatedconv1(x)
        x = self.deconv1(x)
        x = self.conv2(x)
        return self.sigmoid(x)

if __name__ == "__main__":
    model = USNet()

    # Create a random input tensor
    x = torch.randn((2, 3, 512, 512))

    # Pass the input tensor through the model
    y = model(x)

    # Print the shape of the output
    print(y.shape)
