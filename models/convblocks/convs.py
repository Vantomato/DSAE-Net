import torch
import torch.nn as nn
import torch.nn.functional as F
from .CotAttention import C2f_CoTAttention
from .CoordAtt import CoordAtt


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool=True):
        '''
        pool_mode can be False (no pooling) or True ('maxpool')
        '''
        super(ConvBlock, self).__init__()
        if shortcut==True: self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm2d(out_c))
        else: self.shortcut=False
        pad = (k_sz - 1) // 2

        block = []
        if pool: self.pool = nn.MaxPool2d(kernel_size=2)
        else: self.pool = False

        block.append(nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        block.append(nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        self.block = nn.Sequential(*block)
    def forward(self, x):
        if self.pool: x = self.pool(x)
        out = self.block(x)
        if self.shortcut: return out + self.shortcut(x)
        else: return out

class ConvBlock_6(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool=True):
        '''
        pool_mode can be False (no pooling) or True ('maxpool')
        '''
        super(ConvBlock_6, self).__init__()
        if shortcut == True:
            self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm2d(out_c))
        else:
            self.shortcut = False
        pad = (k_sz - 1) // 2

        block = []
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=2)
        else:
            self.pool = False
        block.append(C2f_CoTAttention(c1=in_c, c2=out_c))

        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))
        block.append(C2f_CoTAttention(c1=out_c, c2=out_c))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        if self.pool: x = self.pool(x)
        out = self.block(x)
        if self.shortcut:
            return out + self.shortcut(x)
        else:
            return out

class UpsampleBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, up_mode='transp_conv'):
        super(UpsampleBlock, self).__init__()
        block = []
        if up_mode == 'transp_conv':
            block.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))
        elif up_mode == 'up_conv':
            block.append(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False))
            block.append(nn.Conv2d(in_c, out_c, kernel_size=1))
        else:
            raise Exception('Upsampling mode not supported')

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class ConvBridgeBlock(torch.nn.Module):
    def __init__(self, channels, k_sz=3):
        super(ConvBridgeBlock, self).__init__()
        pad = (k_sz - 1) // 2
        block=[]

        block.append(nn.Conv2d(channels, channels, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(channels))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UpConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, up_mode='up_conv', conv_bridge=False, shortcut=False):
        super(UpConvBlock, self).__init__()
        self.conv_bridge = conv_bridge

        self.up_layer = UpsampleBlock(in_c, out_c, up_mode=up_mode)
        self.conv_layer = ConvBlock(2 * out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False)
        if self.conv_bridge:
            self.conv_bridge_layer = ConvBridgeBlock(out_c, k_sz=k_sz)

    def forward(self, x, skip):
        up = self.up_layer(x)
        if self.conv_bridge:
            out = torch.cat([up, self.conv_bridge_layer(skip)], dim=1)
        else:
            out = torch.cat([up, skip], dim=1)
        out = self.conv_layer(out)
        return out

class UpConvBlock_7(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, up_mode='up_conv', conv_bridge=False, shortcut=False):
        super(UpConvBlock_7, self).__init__()
        self.conv_bridge = conv_bridge

        self.up_layer = UpsampleBlock(in_c, out_c, up_mode=up_mode)
        self.conv_layer = ConvBlock(2 * out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False)
        if self.conv_bridge:
            self.conv_bridge_layer = CoordAtt(out_c, out_c, 32)

    def forward(self, x, skip):
        up = self.up_layer(x)
        if self.conv_bridge:
            out = torch.cat([up, self.conv_bridge_layer(skip)], dim=1)
        else:
            out = torch.cat([up, skip], dim=1)
        out = self.conv_layer(out)
        return out

class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.register_buffer('drop_prob', drop_prob * torch.ones(1, dtype=torch.float32))
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask and place on input device
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).to(x)

            # compute block mask
            block_mask, keeped = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * (block_mask.numel() / keeped).to(out)
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        keeped = block_mask.numel() - block_mask.sum().to(torch.float32)  # prevent overflow in float16
        block_mask = 1 - block_mask.squeeze(1)

        return block_mask, keeped

    def _compute_gamma(self, x):
        return self.drop_prob.item() / (self.block_size ** 2)

class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super(SpatialAttention, self).__init__()
        self.squeeze = nn.Conv2d(dim, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ChannelAttention, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(dim // reduction, dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z

if __name__ == '__main__':
    block = UpConvBlock(in_c=32, out_c=64, k_sz=3, up_mode='transp_conv', conv_bridge=True, shortcut=True)
    input = torch.rand(1, 32, 512, 512).to("cuda")
    block = block.to("cuda")
    output = block(input)
    parms = sum(p.numel() for p in block.parameters())
    print('parameters:{}'.format(parms))
    print(input.size())
    print(output.size())