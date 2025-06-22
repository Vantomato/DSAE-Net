
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class DropBlock2D(nn.Module):
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
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 使用He初始化（对应Keras的he_normal）
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='conv2d')

    def forward(self, x):
        # 输入x的shape: (batch, channel, height, width)

        # 通道维度的平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, h, w)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (batch, 1, h, w)

        # 拼接通道特征
        concat = torch.cat([avg_out, max_out], dim=1)  # (batch, 2, h, w)

        # 空间注意力权重
        sa = self.conv(concat)  # (batch, 1, h, w)
        sa = self.sigmoid(sa)

        # 应用注意力权重
        return x * sa  # (batch, c, h, w)

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            DropBlock2D(block_size=7, drop_prob=0.1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            DropBlock2D(block_size=7, drop_prob=0.1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_little(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block_little, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            DropBlock2D(block_size=7, drop_prob=0.1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x



class SA_UNet(nn.Module):
    # # in_channels=in_c, num_classes
    def __init__(self, in_channels=3, num_classes=1):
        super(SA_UNet, self).__init__()
        in_ch = in_channels
        out_ch = num_classes
        self.n_classes = out_ch
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])

        self.Convm = nn.Sequential(
            conv_block_little(filters[2], filters[3]),
            SpatialAttention(),
            conv_block_little(filters[3], filters[3]),
        )

        self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        f = self.Maxpool3(e3)
        f = self.Convm(f)

        d3 = self.Up4(f)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.Up_conv4(d3)

        d2 = self.Up3(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.Up_conv3(d2)

        d1 = self.Up2(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.Up_conv2(d1)

        out = self.Conv(d1)
        return out

if __name__ == '__main__':
    input = torch.randn((1, 3, 512, 512))
    model = SA_UNet(in_channels=3, num_classes=2)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    output = model(input)
    print(output.shape)

