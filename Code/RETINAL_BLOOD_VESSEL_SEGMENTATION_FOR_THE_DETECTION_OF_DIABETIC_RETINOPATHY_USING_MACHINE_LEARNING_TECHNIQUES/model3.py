import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c,momentum=0.99)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c,momentum=0.99)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(p=0.01)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x= self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x= self.relu(x)
        x=self.dropout(x)
        p = self.pool(x)
        return x, p

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = ConvBlock(in_channels, in_channels)
        self.attention_map = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x,_=self.conv(x) 
        f=self.attention_map(x)
        return x*f

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels,momentum=0.99)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.01)

    def forward(self, x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        x=self.dropout(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2,padding=0)
        self.bn = nn.BatchNorm2d(out_channels,momentum=0.99)
        self.relu = nn.ReLU(inplace=True)
        self.conv = ConvBlock(out_channels+out_channels, out_channels)
    def forward(self, x,p):
        x= self.relu(self.bn(self.deconv(x)))
        x=torch.cat([x,p],axis=1)
        x,_=self.conv(x)
        return x

class USNet(nn.Module):
    def __init__(self):
        super(USNet, self).__init__()
        self.conv1 =ConvBlock(3, 64)
        self.conv2=ConvBlock(64, 128)
        self.conv3=ConvBlock(128,256)
        self.conv4=ConvBlock(256,512)
        self.att1 =AttentionBlock(128)
        self.att2=AttentionBlock(512)
        self.dilatedconv1 =DilatedConvBlock(512, 1024, 2)
        #self.conv6=ConvBlock(512,1024)
        self.dilatedconv2=DilatedConvBlock(1024, 1024, 2)
        self.att3 =AttentionBlock(256)
        self.att4=AttentionBlock(1024)
        self.att5=AttentionBlock(1024)
        self.deconv1 =DeconvBlock(1024, 512)
        self.deconv2 =DeconvBlock(512, 256)
        self.deconv3=DeconvBlock(256, 128)
        self.deconv4=DeconvBlock(128, 64)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=1,padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1,p1 = self.conv1(x)
        x2,p2=self.conv2(p1)
        x = self.att1(p2)
        x3,p3=self.conv3(x)
        x=self.att3(p3)
        x4,p4=self.conv4(x)
        x=self.att2(p4)
        x = self.dilatedconv1(x)
        #x5,_=self.conv6(x)
        x = self.dilatedconv2(x)
        x=self.att4(x)
        x=self.att5(x)
        x = self.deconv1(x,x4)
        x = self.deconv2(x,x3)
        x = self.deconv3(x,x2)
        x = self.deconv4(x,x1)
        x = self.conv5(x)
        return self.sigmoid(x)

if __name__ == "__main__":
    model = USNet()
    # Create a random input tensor
    x = torch.randn((2, 3, 512, 512))
    # Pass the input tensor through the model
    y = model(x)
    # Print the shape of the output
    print(y.shape)
