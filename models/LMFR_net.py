import torch
import torch.nn as nn
from thop import profile
from torch.nn import functional as func


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Conv2d(out_channels * 2, out_channels, kernel_size=(1, 1), padding=0, bias=False)

    def forward(self, x):
        out = self.conv_out(torch.cat([self.conv1(x), self.conv2(x)], dim=1))
        #out = self.conv2(x)
        return out

class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        # self.down = nn.MaxPool2d((2, 2), stride=2)
        self.down = nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False, padding=0)
        # self.conv = Conv_Block(in_channels, out_channels)
        
    def forward(self, x):
        d_x = self.down(x)
        # out = self.conv(d_x)
        return d_x
        

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)
        self.conv = Conv_Block(in_channels * 2, out_channels)

    def forward(self, x, feature_map):
        u_x = self.up(x)
        # feature_map = crop_img(feature_map, out)
        diff_y = feature_map.size()[2] - u_x.size()[2]
        diff_x = feature_map.size()[3] - u_x.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        u_x = func.pad(u_x, [diff_x // 2, diff_x - diff_x // 2,
                             diff_y // 2, diff_y - diff_y // 2])
        c_x = torch.cat((feature_map, u_x), dim=1)
        out = self.conv(c_x)
        return out
        
class UpSample_d2_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample_d2_1, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)

    def forward(self, x):
        u_x = self.up(x)

        return u_x
        
        
class UpSample_d2_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample_d2_2, self).__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)
        self.up2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=(2, 2), stride=2)

    def forward(self, x):
        u_x = self.up2(self.up1(x))

        return u_x
        
        
class SE_Block(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(
            nn.Linear(channels, channels * 2 // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(channels * 2 // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, X_input):
        b, c, _, _ = X_input.size()  # shape = [16, 576, 24, 24]

        y = self.avg_pool(X_input)  # shape = [16, 576, 1, 1]
        y = y.view(b, c)  # shape = [16, 576]

        y = self.linear1(y)  # shape = [16, 576] * [576, 36] = [16, 36]

        y = self.linear2(y)  # shape = [16, 36] * [36, 576] = [16, 576]
        y = y.view(b, c, 1, 1)  

        return X_input * y.expand_as(X_input)


class LMAFF(nn.Module):
    def __init__(self, in_channels, out_channels, d_rate1, d_rate2):
        super(LMAFF, self).__init__()
        self.conv_path1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), dilation=d_rate1, padding=d_rate1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), dilation=d_rate2, padding=d_rate2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7,
                              padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self.se = SE_Block(out_channels * 2)

        self.conv_out = nn.Conv2d(out_channels * 2, out_channels, kernel_size=(1, 1), padding=0, bias=False)
        

    def forward(self, x):
        # spatial attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))

        path1 = self.conv_path1(x)
        path2 = self.conv_path2(x)
        
        out = torch.cat((spatial_out * path1, spatial_out * path2), dim=1)
        out = self.conv_out(self.se(out))
        return out


class FEB(nn.Module):
    def __init__(self, in_c, channels):
        super(FEB, self).__init__()
        self.ch_conv = nn.Conv2d(in_c, channels, kernel_size=(1, 1), padding=0)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.active = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(self.active(self.ch_conv(x)))
        x1 = self.active(x + self.conv(x))
        x2 = self.active(x + x1 + self.conv(x1))
        x3 = self.active(x + x1 + x2 + self.conv(x2))
        x4 = self.active(x + x1 + x2 + x3 + self.conv(x3))
        
        return self.conv(self.conv(x4))
        

class LMFR_Net(nn.Module):
    def __init__(self,
                 in_channels = 3,
                 num_classes = 2,
                 base_c = 16):
        super(LMFR_Net, self).__init__()
        # in_channels=in_c, num_classes
        self.n_classes = num_classes
        # encoder
        self.e1 = Conv_Block(in_channels, base_c)
        self.down = DownSample(base_c)
        self.e2 = Conv_Block(base_c, base_c)
        #self.d1_2 = DownSample()
        self.e3 = Conv_Block(base_c, base_c)
        
        # skip connection
        self.skip1 = LMAFF(base_c, base_c, 9, 7)
        self.skip2 = LMAFF(base_c, base_c, 7, 5)
        self.skip3 = LMAFF(base_c, base_c, 5, 3)
        self.feb = FEB(in_channels, base_c)
        
        # decoder_1
        self.u1_1 = UpSample(base_c, base_c)
        self.u1_2 = UpSample(base_c, base_c)
        # decoder_2
        self.d2_1 = Conv_Block(base_c, base_c)
        self.u2_1 = UpSample_d2_2(base_c, base_c)
        self.d2_2 = Conv_Block(base_c, base_c)
        self.u2_2 = UpSample_d2_1(base_c, base_c)
        self.d2_3 = Conv_Block(base_c, base_c)
        
        # out
        # self.conv_squeeze = nn.Conv2d(base_c * 3, base_c, kernel_size=(1, 1), padding=0)
        self.out_conv = nn.Conv2d(base_c, num_classes, kernel_size=(1, 1), padding=0)
        
        self.active = nn.Sigmoid()

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.down(x1))
        x3 = self.e3(self.down(x2))

        x4 = self.u1_1(self.skip3(x3), self.skip2(x2))
        x5 = self.u1_2(x4, self.skip1(x1))

        x6 = self.d2_1(x3)  # 64->32
        x7 = self.d2_2(x4)  # 32->16
        x8 = self.d2_3(x5)
        
        f1 = self.feb(x)
        
        #########################################################################
        c2 = self.u2_2(x7)  # 16->16
        c3 = self.u2_1(x6)  # 32->16
        # feature_map = crop_img(feature_map, out)
        diff_y1 = x8.size()[2] - c2.size()[2]
        diff_x1 = x8.size()[3] - c2.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        c2 = func.pad(c2, [diff_x1 // 2, diff_x1 - diff_x1 // 2,
                             diff_y1 // 2, diff_y1 - diff_y1 // 2])
        # feature_map = crop_img(feature_map, out)
        diff_y2 = x8.size()[2] - c3.size()[2]
        diff_x2 = x8.size()[3] - c3.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        c3 = func.pad(c3, [diff_x2 // 2, diff_x2 - diff_x2 // 2,
                             diff_y2 // 2, diff_y2 - diff_y2 // 2])
        # feature_map = crop_img(feature_map, out)
        diff_y3 = x8.size()[2] - f1.size()[2]
        diff_x3 = x8.size()[3] - f1.size()[3]
        # padding_left, padding_right, padding_top, padding_bottom
        f1 = func.pad(f1, [diff_x3 // 2, diff_x3 - diff_x3 // 2,
                             diff_y3 // 2, diff_y3 - diff_y3 // 2])
                             
        #########################################################################
                             
        # x9 = torch.cat([x8, c2, c3], dim=1)
        out1 = self.active(x8)
        out2 = self.active(c2)
        out3 = self.active(c3)
        out_feb = self.active(f1)
        out = 0.5 * out1 + 0.2 * out2 + 0.2 * out3 + 0.1 * out_feb

        out = self.out_conv(out)

        return out


if __name__ == '__main__':
    input = torch.randn((1, 4, 512, 512))
    model = LMFR_Net(in_channels=4, num_classes=2)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    output = model(input)
    print(output.shape)



