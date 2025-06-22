import os, sys
import os.path as osp
import argparse
import warnings
import time

import pandas as pd

from utils import paired_transforms_tv04 as p_tr
from PIL import Image
from skimage.io import imsave
from skimage.util import img_as_ubyte
from skimage.transform import resize
import torch
from get_model import get_arch
from utils.model_saving_loading import load_model

# argument parsing
parser = argparse.ArgumentParser()
required_named = parser.add_argument_group('required arguments')
parser.add_argument('--model_path', help='experiments/subfolder where checkpoint is', default='experiments/wnet_drive1')
parser.add_argument('--im_path', help='path to image to be segmented', default=None)
parser.add_argument('--mask_path', help='path to FOv mask, will be computed if not provided', default=None)
parser.add_argument('--tta', type=str, default='from_preds', help='test-time augmentation (no/from_logits/from_preds)')
parser.add_argument('--bin_thresh', type=float, default='0.4196', help='binarizing threshold')
# im_size overrides config file
parser.add_argument('--im_size', help='delimited list input, could be 600,400', type=str, default='512')
parser.add_argument('--device', type=str, default='cuda:0', help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')
parser.add_argument('--result_path', type=str, default=None, help='path to save prediction)')

from skimage import measure, draw
import numpy as np
from torchvision.transforms import Resize
from scipy import optimize
from skimage.filters import threshold_minimum
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes
from skimage.exposure import equalize_adapthist

def get_circ(binary):
    # https://stackoverflow.com/a/28287741
    image = binary.astype(int)
    regions = measure.regionprops(image)
    bubble = regions[0]

    y0, x0 = bubble.centroid
    r = bubble.major_axis_length / 2.

    def cost(params):
        x0, y0, r = params
        coords = draw.circle(y0, x0, r, shape=image.shape)
        template = np.zeros_like(image)
        template[coords] = 1
        return -np.sum(template == image)

    x0, y0, r = optimize.fmin(cost, (x0, y0, r))
    return x0, y0, r

def create_circular_mask(sh, center=None, radius=None):
    # https://stackoverflow.com/a/44874588
    h, w = sh
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def get_fov(img):
    im_s = img.size
    if max(im_s) > 500:
        img = Resize(500)(img)

    with np.errstate(divide='ignore'):
        im_v = equalize_adapthist(np.array(img))[:, :, 1]
        # im_v = equalize_adapthist(rgb2hsv(np.array(img))[:, :, 2])
    thresh = threshold_minimum(im_v)
    binary = binary_fill_holes(im_v > thresh)

    x0, y0, r = get_circ(binary)
    fov = create_circular_mask(binary.shape, center=(x0, y0), radius=r)

    return Resize(im_s[ : :-1])(Image.fromarray(fov))

def crop_to_fov(img, mask):
    mask = np.array(mask).astype(int)
    minr, minc, maxr, maxc = regionprops(mask)[0].bbox
    im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
    return im_crop, [minr, minc, maxr, maxc]

def flip_ud(tens):
    return torch.flip(tens, dims=[1])

def flip_lr(tens):
    return torch.flip(tens, dims=[2])

def flip_lrud(tens):
    return torch.flip(tens, dims=[1, 2])

def create_pred(model, tens, mask, coords_crop, original_sz, bin_thresh, tta='no'):
    act = torch.sigmoid if model.n_classes == 1 else torch.nn.Softmax(dim=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        logits = model(tens.unsqueeze(dim=0).to(device)).squeeze(dim=0)
    pred = act(logits)

    if tta!='no':
        with torch.no_grad():
            logits_lr = model(tens.flip(-1).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-1)
            logits_ud = model(tens.flip(-2).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-2)
            logits_lrud = model(tens.flip(-1).flip(-2).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-1).flip(-2)

        if tta == 'from_logits':
            mean_logits = torch.mean(torch.stack([logits, logits_lr, logits_ud, logits_lrud]), dim=0)
            pred = act(mean_logits)
        elif tta == 'from_preds':
            pred_lr = act(logits_lr)
            pred_ud = act(logits_ud)
            pred_lrud = act(logits_lrud)
            pred = torch.mean(torch.stack([pred, pred_lr, pred_ud, pred_lrud]), dim=0)
        else: raise NotImplementedError
    pred = pred.detach().cpu().numpy()[-1]  # this takes last channel in multi-class, ok for 2-class
    # Orders: 0: NN, 1: Bilinear(default), 2: Biquadratic, 3: Bicubic, 4: Biquartic, 5: Biquintic
    pred = resize(pred, output_shape=original_sz, order=3)
    full_pred = np.zeros_like(mask, dtype=float)
    full_pred[coords_crop[0]:coords_crop[2], coords_crop[1]:coords_crop[3]] = pred
    full_pred[~mask.astype(bool)] = 0
    full_pred_bin = full_pred > bin_thresh
    return full_pred, full_pred_bin


def pred_item(config):
    if config.device.startswith("cuda"):
        # In case one has multiple devices, we must first set the one
        # we would like to use so pytorch can find it.
        os.environ['CUDA_VISIBLE_DEVICES'] = config.device.split(":",1)[1]
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is not currently available!")
        print(f"* Running prediction on device '{config.device}'...")
        device = torch.device("cuda")
    else:  #cpu
        device = torch.device(config.device)

    bin_thresh = config.bin_thresh
    tta = config.tta

    model_name = config.model_name
    model_path = config.model_path
    im_path = config.im_path
    im_loc = osp.dirname(im_path)
    im_name = im_path.rsplit('/', 1)[-1]

    mask_path = config.mask_path
    result_path = config.result_path
    if result_path is None:
        result_path = im_loc
        im_path_out = osp.join(result_path, im_name.rsplit('.', 1)[-2]+'_seg.png')
        im_path_out_bin = osp.join(result_path, im_name.rsplit('.', 1)[-2]+'_bin_seg.png')
    else:
        os.makedirs(result_path, exist_ok=True)
        im_path_out = osp.join(result_path, im_name.rsplit('.', 1)[-2]+'_seg.png')
        im_path_out_bin = osp.join(result_path, im_name.rsplit('.', 1)[-2] + '_bin_seg.png')

    im_size = tuple([int(item) for item in config.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    print('* Segmenting image ' + im_path)
    img = Image.open(im_path)
    if mask_path is None:
        print('* FOV mask not provided, generating it')
        mask = get_fov(img)
        print('* FOV mask generated')
    else: mask = Image.open(mask_path).convert('L')
    mask = np.array(mask).astype(bool)

    img, coords_crop = crop_to_fov(img, mask)
    original_sz = img.size[1], img.size[0]  # in numpy convention

    rsz = p_tr.Resize(tg_size)
    tnsr = p_tr.ToTensor()
    tr = p_tr.Compose([rsz, tnsr])
    im_tens = tr(img)  # only transform image

    print('* Instantiating model  = ' + str(model_name))
    model = get_arch(model_name).to(device)
    if model_name == 'wnet': model.mode='eval'
    model.mode = 'eval'

    print('* Loading trained weights from ' + model_path)
    model, stats = load_model(model, model_path, device)
    model.eval()

    print('* Saving prediction to ' + im_path_out)
    start_time = time.perf_counter()
    full_pred, full_pred_bin = create_pred(model, im_tens, mask, coords_crop, original_sz, bin_thresh=bin_thresh, tta=tta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # imsave(im_path_out, img_as_ubyte(full_pred))
        # imsave(im_path_out_bin, img_as_ubyte(full_pred_bin))
    return img_as_ubyte(full_pred_bin),img_as_ubyte(full_pred)
    # print('Done, time spent = {:.3f} secs'.format(time.perf_counter() - start_time))

def get_model_path(model_name,dataset):
    path = rf"D:\study\experiment\lwnet-master\experiments/{model_name}_{dataset}/"

    return path




def pred_all():
    pred_all_path = r"D:\study\experiment\lwnet-master\paperresults/pred_all/"
    dataset_arr = ['DRIVE', 'STARE', 'CHASEDB1', 'HRF']
    model_arr = ['U_Net', 'My76_WNet', 'AttU_Net', 'SA_UNet', 'ConvUNeXt', 'ULite']
    # 模型路径名称（用于读取）
    model_read_names = ['U_Net', 'AttU_Net', 'SA_UNet', 'ConvUNeXt', 'ULite', 'My76_WNet']

    for dataset in dataset_arr:
        if not os.path.exists(pred_all_path + dataset):
            os.makedirs(pred_all_path + dataset)

        # 读取数据集的csv文件
        dataset_csv = 'data/' + dataset + '/test_all.csv'
        df = pd.read_csv(dataset_csv)
        im_paths = df['im_paths']
        # 使用更可靠的方式提取文件名
        im_names = [os.path.splitext(os.path.basename(p))[0] for p in im_paths]
        for model in model_arr:
            if not os.path.exists(pred_all_path + dataset + '/' + model):
                os.makedirs(pred_all_path + dataset + '/' + model)
            # 新建config
            config = argparse.Namespace()
            config.model_path = get_model_path(model,dataset)
            config.tta = 'from_preds'
            config.im_size = '512'
            config.result_path = pred_all_path + dataset + '/' + model + '/'
            config.model_name = model
            config.bin_thresh = 0.4196
            config.device = 'cuda:0'
            for idx, row in df.iterrows():
                # 读取图像和掩膜路径
                im_path = row['im_paths']
                mask_path = row['gt_paths']
                config.im_path = im_path
                config.mask_path = mask_path
                # 预测
                img,_ = pred_item(config)
                # 保存预测结果
                im_name = os.path.splitext(os.path.basename(im_path))[0]
                im_path_out_bin = os.path.join(config.result_path, im_name + '_bin_seg.png')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imsave(im_path_out_bin, img)
                print(f"Saved prediction for {im_name} to {im_path_out_bin}")






    # 对每个数据集的每张图用每个模型进行预测，然后得到二分类的full_pred_bin图，将得到的图保存到对应的文件夹中，

def statistics_pred():
    """
    统计对应的文件夹中的图片，将它们的二分类结果进行统计，取得每个数据集上的每个模型在每张图上的mcc作为一个dataframe
    Returns:

    """

    return None
if __name__ == '__main__':

    args = parser.parse_args()

    if True:
        args.model_path = 'experiments/My76_WNet_DRIVE/'
        args.im_path = 'data/DRIVE/all_images/01_test.tif'
        args.mask_path = 'data/DRIVE/all_mask/01_test_mask.gif'
        args.tta = 'from_preds'
        args.im_size = '512'
        args.result_path = 'results/DRIVE'
        args.model_name = 'My76_WNet'
        args.bin_thresh = 0.4196

    if args.device.startswith("cuda"):
        # In case one has multiple devices, we must first set the one
        # we would like to use so pytorch can find it.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(":",1)[1]
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is not currently available!")
        print(f"* Running prediction on device '{args.device}'...")
        device = torch.device("cuda")
    else:  #cpu
        device = torch.device(args.device)

    bin_thresh = args.bin_thresh
    tta = args.tta

    model_name = args.model_name
    model_path = args.model_path
    im_path = args.im_path
    im_loc = osp.dirname(im_path)
    im_name = im_path.rsplit('/', 1)[-1]

    mask_path = args.mask_path
    result_path = args.result_path
    if result_path is None:
        result_path = im_loc
        im_path_out = osp.join(result_path, im_name.rsplit('.', 1)[-2]+'_seg.png')
        im_path_out_bin = osp.join(result_path, im_name.rsplit('.', 1)[-2]+'_bin_seg.png')
    else:
        os.makedirs(result_path, exist_ok=True)
        im_path_out = osp.join(result_path, im_name.rsplit('.', 1)[-2]+'_seg.png')
        im_path_out_bin = osp.join(result_path, im_name.rsplit('.', 1)[-2] + '_bin_seg.png')

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    print('* Segmenting image ' + im_path)
    img = Image.open(im_path)
    if mask_path is None:
        print('* FOV mask not provided, generating it')
        mask = get_fov(img)
        print('* FOV mask generated')
    else: mask = Image.open(mask_path).convert('L')
    mask = np.array(mask).astype(bool)

    img, coords_crop = crop_to_fov(img, mask)
    original_sz = img.size[1], img.size[0]  # in numpy convention

    rsz = p_tr.Resize(tg_size)
    tnsr = p_tr.ToTensor()
    tr = p_tr.Compose([rsz, tnsr])
    im_tens = tr(img)  # only transform image

    print('* Instantiating model  = ' + str(model_name))
    model = get_arch(model_name).to(device)
    if model_name == 'wnet': model.mode='eval'
    model.mode = 'eval'

    print('* Loading trained weights from ' + model_path)
    model, stats = load_model(model, model_path, device)
    model.eval()

    print('* Saving prediction to ' + im_path_out)
    start_time = time.perf_counter()
    full_pred, full_pred_bin = create_pred(model, im_tens, mask, coords_crop, original_sz, bin_thresh=bin_thresh, tta=tta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(im_path_out, img_as_ubyte(full_pred))
        imsave(im_path_out_bin, img_as_ubyte(full_pred_bin))
    print('Done, time spent = {:.3f} secs'.format(time.perf_counter() - start_time))



