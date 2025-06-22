import sys
import time
from thop import profile
from models import (AttU_Net, ConvUNeXt, FR_UNet, LMFR_Net,PDFUNet, ResUNetPlusPlus, SA_UNet, ULite, My76_WNet, UNettp)
import torch

def get_arch(model_name, in_c=3, n_classes=1):
    if model_name == 'AttU_Net':
        model = AttU_Net(in_channels=in_c, num_classes=n_classes)
    elif model_name == 'ConvUNeXt':
        model = ConvUNeXt(in_channels=in_c, num_classes=n_classes)
    elif model_name == 'FR_UNet':
        model = FR_UNet(in_channels=in_c, num_classes=n_classes)
    elif model_name == 'LMFR_Net':
        model = LMFR_Net(in_channels=in_c, num_classes=n_classes)
    elif model_name == 'PDFUNet':
        model = PDFUNet(in_channels=in_c, num_classes=n_classes)
    elif model_name == 'ResUNetPlusPlus':
        model = ResUNetPlusPlus(in_channels=in_c, num_classes=n_classes)
    elif model_name == 'SA_UNet':
        model = SA_UNet(in_channels=in_c, num_classes=n_classes)
    elif model_name == 'ULite':
        model = ULite(in_channels=in_c, num_classes=n_classes)
    elif model_name == 'My76_WNet':
        model = My76_WNet(in_c=in_c, n_classes=n_classes, layers=[8,16,32], conv_bridge=True, shortcut=True)
    elif model_name == 'UNettp':
        model = UNettp(in_channels=in_c, num_classes=n_classes)
    else: sys.exit('not a valid model_name, check models.get_model.py')
    return model

def calculate_flops_params_infer(model):
    input = torch.randn((1, 3, 512, 512))
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2) ) + 'G')
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + 'M')
    start_time = time.time()
    test_time = 10
    for i in range(test_time):
        output = model(input)
    Inference = str(round((time.time() - start_time)/test_time, 2))
    print('Inference time = ' + Inference + 's')

    return flops, params

def calculate_all(model_name_arr):
    for model_name in model_name_arr:
        print('model_name = ' + model_name)
        model = get_arch(model_name, in_c=3, n_classes=1)
        model.mode = 'eval'
        calculate_flops_params_infer(model)

if __name__ == '__main__':
    model_name = 'My76_WNet'
    input = torch.randn((1, 3, 512, 512)).to('cuda')
    model = get_arch(model_name, in_c=3, n_classes=1).to('cuda')
    model.mode='eval'
    model.eval()
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    output = model(input)
    print(output.shape)