import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from options import config
from network.vit_seg_modeling import VisionTransformer as ViT_seg
from network.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from .vgg import VGG

class BasicConv2dReLu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2dReLu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = max_out
        return self.sigmoid(out)

class ChannelAttentionJoin(nn.Module): # input channel is 2C, while output channel is C
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttentionJoin, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes // 2, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_5, x_cur):
        # can add the channel_shuffle to torch.cat(self.avg_pool(x_5),self.avg_pool(x_cur))
        max_out = self.fc2(self.relu1(self.fc1(torch.cat((self.avg_pool(x_5),self.avg_pool(x_cur)), 1))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class TRU(nn.Module):
    def __init__(self, channel):
        super(TRU, self).__init__()

        self.query_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.key_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.value_conv_cur = nn.Conv2d(channel, channel, kernel_size=1)

        self.gamma_cur = nn.Parameter(torch.ones(1))
        # following DANet
        self.conv_cur = nn.Sequential(BasicConv2dReLu(channel, channel, 3, padding=1),
                                    nn.Dropout2d(0.1, False),
                                    BasicConv2dReLu(channel, channel, 1)
                                    )

    def forward(self, x_1, x_cur): # x_1: Q, x_cur: K
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: C X H x W
        """
        proj_query = self.query_conv(x_1) # B X C X H2 x W2
        proj_key = self.key_conv(x_cur) # B X C X H1 x W1
        proj_query_t = torch.transpose(proj_query,2,3).contiguous()  # B X C X W2 x H2 (W=H)
        energy = torch.matmul(proj_query_t, proj_key) # C X (W2 x H2) x (H1 X W1) = C X W2 x W1
        attention = self.softmax(energy) # C X W2 x W1

        proj_value_cur = self.value_conv_cur(x_cur)  # C X H1 x W1

        out_cur = torch.matmul(proj_value_cur, attention).contiguous()  # C X H1 x W1 X (W2 x W1) = C X H1 X W1
        out_cur = self.conv_cur(self.gamma_cur * out_cur + x_cur)

        return out_cur

class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2dReLu(4*out_channel, in_channel, 3, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        return x_cat


class TSCM(nn.Module):
    def __init__(self, cur_channel):
        super(TSCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.ca_join = ChannelAttentionJoin(2*cur_channel)
        self.sa = SpatialAttention()
        self.conv = BasicConv2dReLu(cur_channel, cur_channel, 3, padding=1, dilation=1)

        self.TRU = TRU(cur_channel)

        self.msp = RFB(cur_channel, cur_channel//4)

        config_vit = CONFIGS_ViT_seg[config.vit_name]

        self.net = ViT_seg(config_vit, img_size=config.img_size, in_channels=cur_channel).cuda()
        self.net.load_from(weights=np.load(config_vit.pretrained_path))
        self.down = nn.AdaptiveAvgPool2d((32,32))

        self.fuse_conv = BasicConv2dReLu(2*cur_channel, cur_channel, 3, padding=1, dilation=1)

    def forward(self, x_1, x_cur, x_5):
        # PAU
        # Join CA
        x_cur_ca = x_cur.mul(self.ca_join(x_5,x_cur))
        # SA
        fea_size = x_cur.size()[2:]
        x_5_up = F.interpolate(x_5, size=fea_size, mode="bilinear", align_corners=True)
        x_cur_ca_sa = x_cur_ca.mul(self.sa(x_5_up))
        # st means shortcut
        x_cur_casa_st = x_cur_ca_sa+x_cur


        # TRU
        x_cur_casa_up = self.conv(self.upsample2(x_cur_casa_st))
        x_1_down = F.interpolate(x_1, size=[fea_size[0]*2, fea_size[1]*2], mode="bilinear", align_corners=True)
        x_cur_detail = self.TRU(x_1_down, x_cur_casa_up)


        # RIU
        # Multi-scale perception
        x_cur_msp =  self.msp(x_cur_ca+x_cur)
        # ViT
        x_cur_tf = self.net(self.down(x_cur_msp))
        x_cur_tf = F.interpolate(x_cur_tf, size=[fea_size[0]*2, fea_size[1]*2], mode="bilinear", align_corners=True)


        x_out = self.fuse_conv(torch.cat((x_cur_tf, x_cur_detail), 1))

        return x_out


class decoder(nn.Module):
    def __init__(self, channel):
        super(decoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S4 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)

        self.decoder3 = nn.Sequential(
            BasicConv2d(2*channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S3 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)

        self.decoder2 = nn.Sequential(
            BasicConv2d(2*channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
        )
        self.S2 = nn.Conv2d(channel, 1, 3, stride=1, padding=1)

    def forward(self, x4, x3, x2):
        # x4: 1/4; x3: 1/2; x2: 1/1

        x4_up = self.decoder4(x4)
        s4 = self.S4(x4_up)

        x3_up = self.decoder3(torch.cat((x3, x4_up), 1))
        s3 = self.S3(x3_up)

        x2_up = self.decoder2(torch.cat((x2, x3_up), 1))
        s2 = self.S2(x2_up)

        return s2, s3, s4

class TSCNet(nn.Module):
    def __init__(self, channel=32): # channel can be reduced to 32
        super(TSCNet, self).__init__()
        #Backbone model
        self.vgg = VGG('rgb')

        # input 3x256x256
        self.ChannelNormalization_1 = BasicConv2dReLu(64, channel, 3, 1, 1)  # 64x256x256->64x256x256
        self.ChannelNormalization_2 = BasicConv2dReLu(128, channel, 3, 1, 1) # 128x128x128->64x128x128
        self.ChannelNormalization_3 = BasicConv2dReLu(256, channel, 3, 1, 1) # 256x64x64->64x64x64
        self.ChannelNormalization_4 = BasicConv2dReLu(512, channel, 3, 1, 1) # 512x32x32->64x32x32
        self.ChannelNormalization_5 = BasicConv2dReLu(512, channel, 3, 1, 1) # 512x32x32->64x32x32

        self.TSCM4 = TSCM(channel)
        self.TSCM3 = TSCM(channel)
        self.TSCM2 = TSCM(channel)

        self.decoder_rgb = decoder(channel)

        self.sigmoid = nn.Sigmoid()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x_rgb):
        x1_rgb = self.vgg.conv1(x_rgb)
        x2_rgb = self.vgg.conv2(x1_rgb)
        x3_rgb = self.vgg.conv3(x2_rgb)
        x4_rgb = self.vgg.conv4(x3_rgb)
        x5_rgb = self.vgg.conv5(x4_rgb)

        x1_nor = self.ChannelNormalization_1(x1_rgb)  # 64x256x256
        x2_nor = self.ChannelNormalization_2(x2_rgb)  # 64x128x128
        x3_nor = self.ChannelNormalization_3(x3_rgb)  # 64x64x64
        x4_nor = self.ChannelNormalization_4(x4_rgb)  # 64x32x32
        x5_nor = self.ChannelNormalization_5(x5_rgb)  # 64x32x32

        x4_TSCM = self.TSCM4(x1_nor, x4_nor, x5_nor)
        x3_TSCM = self.TSCM3(x1_nor, x3_nor, x5_nor)
        x2_TSCM = self.TSCM2(x1_nor, x2_nor, x5_nor)

        # s4: 1/2; s3: 1/2; s2: 1/1
        s2, s3, s4 = self.decoder_rgb(x4_TSCM, x3_TSCM, x2_TSCM)

        s4 = self.upsample2(s4)

        return s2, s3, s4, self.sigmoid(s2), self.sigmoid(s3), self.sigmoid(s4)
