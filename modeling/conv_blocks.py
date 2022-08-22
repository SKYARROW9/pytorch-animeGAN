import torch.nn as nn
import torch.nn.functional as F
from utils.common import initialize_weights

class DownConv(nn.Module):

    def __init__(self, channels, bias=False):
        super(DownConv, self).__init__()

        self.conv1 = SeparableConv2D(channels, channels, stride=2, bias=bias)
        self.conv2 = SeparableConv2D(channels, channels, stride=1, bias=bias)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        out2 = self.conv2(out2)

        return out1 + out2


class UpConv(nn.Module):
    def __init__(self, channels, bias=False):
        super(UpConv, self).__init__()

        self.conv = SeparableConv2D(channels, channels, stride=1, bias=bias)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        out = self.conv(out)

        return out


class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super(SeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
            kernel_size=1, stride=1, bias=bias)
        # self.pad = 
        self.ins_norm1 = nn.InstanceNorm2d(in_channels)
        self.activation1 = nn.LeakyReLU(0.2, True)
        self.ins_norm2 = nn.InstanceNorm2d(out_channels)
        self.activation2 = nn.LeakyReLU(0.2, True)

        initialize_weights(self)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.ins_norm1(out)
        out = self.activation1(out)

        out = self.pointwise(out)
        out = self.ins_norm2(out)

        return self.activation2(out)


class ConvBlock(nn.Module):
    def __init__(self, channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.ins_norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

        initialize_weights(self)

    def forward(self, x):
        out = self.conv(x)
        out = self.ins_norm(out)
        out = self.activation(out)

        return out


class InvertedResBlock(nn.Module):
    def __init__(self, channels=256, out_channels=256, expand_ratio=2, bias=False):
        super(InvertedResBlock, self).__init__()
        bottleneck_dim = round(expand_ratio * channels)
        self.conv_block = ConvBlock(channels, bottleneck_dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.depthwise_conv = nn.Conv2d(bottleneck_dim, bottleneck_dim,
            kernel_size=3, groups=bottleneck_dim, stride=1, padding=1, bias=bias)
        self.conv = nn.Conv2d(bottleneck_dim, out_channels,
            kernel_size=1, stride=1, bias=bias)

        self.ins_norm1 = nn.InstanceNorm2d(out_channels)
        self.ins_norm2 = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

        initialize_weights(self)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.depthwise_conv(out)
        out = self.ins_norm1(out)
        out = self.activation(out)
        out = self.conv(out)
        out = self.ins_norm2(out)

        return out + x
    
class GPM(nn.Module): # cvpr19 AFNet

    def __init__(self, in_dim):
        super(GPM, self).__init__()
        down_dim = 512
        n1, n2, n3 = 2, 4, 6
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(down_dim * n1 * n1, down_dim * n1 * n1, kernel_size=3, padding=1),
            nn.BatchNorm2d(down_dim * n1 * n1), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(down_dim * n2 * n2, down_dim * n2 * n2, kernel_size=3, padding=1),
            nn.BatchNorm2d(down_dim * n2 * n2), nn.PReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(down_dim * n3 * n3, down_dim * n3 * n3, kernel_size=3, padding=1),
            nn.BatchNorm2d(down_dim * n3 * n3), nn.PReLU())
        self.fuse = nn.Sequential(nn.Conv2d(3 * down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())

    def forward(self, x):
        conv1 = self.conv1(x)
        ###########################################################################
        gm_2_a = torch.chunk(conv1, 2, 2)
        c = []
        for i in range(len(gm_2_a)):
            b = torch.chunk(gm_2_a[i], 2, 3)
            c.append(torch.cat((b[0], b[1]), 1))
        gm1 = torch.cat((c[0], c[1]), 1)
        gm1 = self.conv2(gm1)
        gm1 = torch.chunk(gm1, 2 * 2, 1)
        d = []
        for i in range(2):
            d.append(torch.cat((gm1[2 * i], gm1[2 * i + 1]), 3))
        gm1 = torch.cat((d[0], d[1]), 2)
        ###########################################################################
        gm_4_a = torch.chunk(conv1, 4, 2)
        e = []
        for i in range(len(gm_4_a)):
            f = torch.chunk(gm_4_a[i], 4, 3)
            e.append(torch.cat((f[0], f[1], f[2], f[3]), 1))
        gm2 = torch.cat((e[0], e[1], e[2], e[3]), 1)
        gm2 = self.conv3(gm2)
        gm2 = torch.chunk(gm2, 4 * 4, 1)
        g = []
        for i in range(4):
            g.append(torch.cat((gm2[4 * i], gm2[4 * i + 1], gm2[4 * i + 2], gm2[4 * i + 3]), 3))
        gm2 = torch.cat((g[0], g[1], g[2], g[3]), 2)
        ###########################################################################
        gm_6_a = torch.chunk(conv1, 6, 2)
        h = []
        for i in range(len(gm_6_a)):
            k = torch.chunk(gm_6_a[i], 6, 3)
            h.append(torch.cat((k[0], k[1], k[2], k[3], k[4], k[5]), 1))

        gm3 = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), 1)
        gm3 = self.conv4(gm3)
        gm3 = torch.chunk(gm3, 6 * 6, 1)
        j = []
        for i in range(6):
            j.append(
                torch.cat((gm3[6 * i], gm3[6 * i + 1], gm3[6 * i + 2], gm3[6 * i + 3], gm3[6 * i + 4], gm3[6 * i + 5]),
                          3))
        gm3 = torch.cat((j[0], j[1], j[2], j[3], j[4], j[5]), 2)
        ###########################################################################

        return self.fuse(torch.cat((gm1, gm2, gm3), 1))
