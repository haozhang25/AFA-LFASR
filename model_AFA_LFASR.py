import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, angRes_in, angRes_out):
        super(Net, self).__init__()
        channels = 64
        n_group = 5
        n_block = 5
        self.angRes = angRes_in
        self.angRes_out = angRes_out

        #################### Initial Feature Extraction #####################
        self.conv_init0 = nn.Sequential(nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False))
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ############# Deep Spatial-Angular Correlation Learning #############
        self.disentg = CascadeDisentgGroup(n_group, n_block, self.angRes, channels)

        ########################### UP-Sampling #############################
        self.upsample = nn.Sequential(
            nn.Conv2d(angRes_in * angRes_in * channels, channels * angRes_out * angRes_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes_out),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, dilation=angRes_out, padding=angRes_out, bias=False)
        )

    def forward(self, x):
        x = rearrange(x, 'b c (u h) (v w) -> b c u v h w', u=self.angRes, v=self.angRes)
        [b, c, u, v, h, w] = x.size()

        x_upscale = LFA_interpolate(x, scale_factor=self.angRes_out / self.angRes, mode='bicubic')
        x_upscale = rearrange(x_upscale, 'b c u v h w -> b c (u h) (v w)', u=self.angRes_out, v=self.angRes_out)

        # Initial Feature Extraction
        x = rearrange(x, 'b c u v h w -> b c (u v) h w', u=self.angRes, v=self.angRes)
        buffer = self.conv_init0(x)
        buffer = self.conv_init(buffer) + buffer

        # Deep Spatial-Angular Correlation Learning
        buffer = rearrange(buffer, "b c (u v) h w -> b c (h u) (w v)", u=self.angRes, v=self.angRes)
        buffer = self.disentg(buffer)

        # UP-Sampling
        buffer = rearrange(buffer, "b c (h u) (w v) -> b (c u v) h w", u=self.angRes, v=self.angRes)
        buffer = self.upsample(buffer)
        buffer = rearrange(buffer, "b c (h u) (w v) -> b c (u h) (v w)", u=self.angRes_out, v=self.angRes_out)
        out = buffer + x_upscale
        return out


class CascadeDisentgGroup(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadeDisentgGroup, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(DisentgGroup(n_block, angRes, channels))
        self.Group = nn.Sequential(*Groups)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                              bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_group):
            buffer = self.Group[i](buffer)
        return self.conv(buffer) + x


class DisentgGroup(nn.Module):
    def __init__(self, n_block, angRes, channels):
        super(DisentgGroup, self).__init__()
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(DisentgBlock(angRes, channels))
        self.Block = nn.Sequential(*Blocks)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                              bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_block):
            buffer = self.Block[i](buffer)
        return self.conv(buffer) + x


class DisentgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DisentgBlock, self).__init__()

        # intra spatial feature
        self.IntraSpaConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False)
        )

        # inter spatial feature
        self.InterSpaConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False)
        )

        # intra angular feature
        self.IntraAngConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), dilation=1,
                      padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(angRes),
        )

        # inter angular feature
        self.InterAngConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=int(angRes * 2), stride=int(angRes * 2),
                      dilation=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, int(4 * angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2 * angRes),
        )

        # horizontal EPI feature
        self.HEPIConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=[1, 2 * angRes + 1], stride=[1, 1], padding=[0, angRes],
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=[1, 2 * angRes + 1], stride=[1, 1], padding=[0, angRes],
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=[1, 2 * angRes + 1], stride=[1, 1], padding=[0, angRes],
                      bias=False)
        )

        # vertical EPI feature
        self.VEPIConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=[2 * angRes + 1, 1], stride=[1, 1], padding=[angRes, 0],
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=[2 * angRes + 1, 1], stride=[1, 1], padding=[angRes, 0],
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=[2 * angRes + 1, 1], stride=[1, 1], padding=[angRes, 0],
                      bias=False)
        )

    def forward(self, buffer):
        feaIntraSpa = self.IntraSpaConv(buffer)
        buffer = feaIntraSpa + buffer
        feaInterSpa = self.InterSpaConv(buffer)
        buffer = feaInterSpa + buffer

        feaIntraAng = self.IntraAngConv(buffer)
        buffer = feaIntraAng + buffer
        feaInterAng = self.InterAngConv(buffer)
        buffer = feaInterAng + buffer

        feaHEPI = self.HEPIConv(buffer)
        buffer = feaHEPI + buffer
        feaVEPI = self.VEPIConv(buffer)
        out = feaVEPI + buffer
        return out

def LFA_interpolate(LF, scale_factor, mode):
    [b, c, u, v, h, w] = LF.size()
    LF = rearrange(LF, 'b c u v h w -> (b h w) c u v')
    LF_upscale = F.interpolate(LF, scale_factor=scale_factor, mode=mode, align_corners=False)
    LF_upscale = rearrange(LF_upscale, '(b h w) c u v -> b c u v h w', h=h, w=w)
    return LF_upscale

if __name__ == "__main__":
    net = Net(angRes_in=2, angRes_out=7).cuda()
    from thop import profile
    input = torch.randn(1, 1, 2*32, 2*32).cuda()
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))