from torch import nn
from models.model_utils import make_layer


class ResidualBlockNoBN(nn.Module):
    def __init__(self, channels=64, ks=3, res_scale=1.0):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, ks, 1, ks//2, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, ks, 1, ks//2, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30, ks=3, stride=1):
        super(ResidualBlocksWithInputConv, self).__init__()

        main = [
            nn.Conv2d(in_channels, out_channels, ks, stride, ks//2, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_blocks, channels=out_channels)
        ]
        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)
