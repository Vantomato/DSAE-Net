"""
对原数据进行处理得到训练集和测试集，可以直接用于训练。或者进行数据增强后再保存。
"""
import os
import shutil
import numpy as np
from torchvision.transforms.functional import resize
import pandas as pd
from PIL import Image


def process_DRIVE(data_path, save_path):
    # 15train 5val 20test
    print('preparing DRIVE data')
    # process drive data, generate CSVs
    data_path = os.path.join(data_path, 'DRIVE')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"file {data_path} does not exists.")
    save_path = os.path.join(save_path, 'DRIVE')
    os.makedirs(save_path, exist_ok=True)
    flag_arr = ['training', 'test']
    choice_arr = ['images', '1st_manual', 'mask']
    for choice in choice_arr:
        dir_path = os.path.join(save_path, 'all_' + choice)
        os.makedirs(dir_path, exist_ok=True)
        for flag in flag_arr:
            if not os.path.exists(os.path.join(data_path, flag, choice)):
                raise FileNotFoundError(f"file {os.path.join(data_path, flag, choice)} does not exists.")
            files = os.listdir(os.path.join(data_path, flag, choice))
            # 将 os.listdir(os.path.join(data_path, flag, choice)) ，目录下的全部文件复制到 os.path.join(save_path, 'all_' + choice) 目录下
            for file in files:
                # 复制文件
                shutil.copy(os.path.join(data_path, flag, choice, file), os.path.join(save_path, 'all_' + choice, file))

    path_ims = os.path.join(save_path, 'all_images')
    path_masks = os.path.join(save_path, 'all_mask')
    path_gts = os.path.join(save_path, 'all_1st_manual')

    all_im_names = sorted(os.listdir(path_ims))
    all_mask_names = sorted(os.listdir(path_masks))
    all_gt_names = sorted(os.listdir(path_gts))

    # append paths
    num_ims = len(all_im_names)
    all_im_names = [os.path.join(path_ims, n) for n in all_im_names]
    all_mask_names = [os.path.join(path_masks, n) for n in all_mask_names]
    all_gt_names = [os.path.join(path_gts, n) for n in all_gt_names]

    test_im_names = all_im_names[:num_ims // 2]
    train_im_names = all_im_names[num_ims // 2:]

    test_mask_names = all_mask_names[:num_ims // 2]
    train_mask_names = all_mask_names[num_ims // 2:]

    test_gt_names = all_gt_names[:num_ims // 2]
    train_gt_names = all_gt_names[num_ims // 2:]

    df_drive_all = pd.DataFrame({'im_paths': all_im_names,
                                 'gt_paths': all_gt_names,
                                 'mask_paths': all_mask_names})

    df_drive_train = pd.DataFrame({'im_paths': train_im_names,
                                   'gt_paths': train_gt_names,
                                   'mask_paths': train_mask_names})

    df_drive_test = pd.DataFrame({'im_paths': test_im_names,
                                  'gt_paths': test_gt_names,
                                  'mask_paths': test_mask_names})

    df_drive_train, df_drive_val = df_drive_train[:15], df_drive_train[15:]

    df_drive_train.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    df_drive_val.to_csv(os.path.join(save_path, 'val.csv'), index=False)
    df_drive_test.to_csv(os.path.join(save_path, 'test.csv'), index=False)
    df_drive_all.to_csv(os.path.join(save_path, 'test_all.csv'), index=False)
    print('DRIVE data prepared')
    return

def process_CHASEDB1(data_path, save_path):
    # 16train 6val 6test?
    print('preparing CHASEDB1 data')
    # process drive CHASEDB1, generate CSVs
    data_path = os.path.join(data_path, 'CHASEDB1')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"file {data_path} does not exists.")
    save_path = os.path.join(save_path, 'CHASEDB1')
    os.makedirs(save_path, exist_ok=True)

    choice_arr = ['images', '1st_label', 'mask']
    for choice in choice_arr:
        dir_path = os.path.join(save_path, 'all_' + choice)
        os.makedirs(dir_path, exist_ok=True)
        files = os.listdir(os.path.join(data_path,'', choice))
        for file in files:
            shutil.copy(os.path.join(data_path, choice, file), os.path.join(save_path, 'all_' + choice, file))

    path_ims = os.path.join(save_path, 'all_images')
    path_masks = os.path.join(save_path, 'all_mask')
    path_gts = os.path.join(save_path, 'all_1st_label')

    all_im_names = sorted(os.listdir(path_ims))
    all_mask_names = sorted(os.listdir(path_masks))
    all_gt_names = sorted(os.listdir(path_gts))

    if len(all_im_names) != len(all_mask_names) or len(all_im_names) != len(all_gt_names):
        raise ValueError('Number of images, masks and ground truths do not match')

    all_im_names = [os.path.join(path_ims, n) for n in all_im_names]
    all_mask_names = [os.path.join(path_masks, n) for n in all_mask_names]
    all_gt_names = [os.path.join(path_gts, n) for n in all_gt_names]

    train_im_names = all_im_names[:22]
    test_im_names = all_im_names[22:]

    train_mask_names = all_mask_names[:22]
    test_mask_names = all_mask_names[22:]

    train_gt_names = all_gt_names[:22]
    test_gt_names = all_gt_names[22:]

    df_chasedb_all = pd.DataFrame({'im_paths': all_im_names,
                                   'gt_paths': all_gt_names,
                                   'mask_paths': all_mask_names})

    df_chasedb_train = pd.DataFrame({'im_paths': train_im_names,
                                     'gt_paths': train_gt_names,
                                     'mask_paths': train_mask_names})

    df_chasedb_test = pd.DataFrame({'im_paths': test_im_names,
                                    'gt_paths': test_gt_names,
                                    'mask_paths': test_mask_names})

    num_ims = len(df_chasedb_train)
    tr_ims = int(0.8 * num_ims)
    tr_ims = 16
    df_chasedb_train, df_chasedb_val = df_chasedb_train[:tr_ims], df_chasedb_train[tr_ims:]

    df_chasedb_train.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    df_chasedb_val.to_csv(os.path.join(save_path, 'val.csv'), index=False)
    df_chasedb_test.to_csv(os.path.join(save_path, 'test.csv'), index=False)
    df_chasedb_all.to_csv(os.path.join(save_path, 'test_all.csv'), index=False)


    print('CHASE-DB prepared')
    return

def process_STARE(data_path, save_path):
    # 12train 4val 4test
    print('preparing STARE data')
    # process drive STARE, generate CSVs
    data_path = os.path.join(data_path, 'STARE')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"file {data_path} does not exists.")
    save_path = os.path.join(save_path, 'STARE')
    os.makedirs(save_path, exist_ok=True)

    choice_arr = ['images', '1st_labels_ah', 'mask']
    for choice in choice_arr:
        dir_path = os.path.join(save_path, 'all_' + choice)
        os.makedirs(dir_path, exist_ok=True)
        files = os.listdir(os.path.join(data_path, choice))
        for file in files:
            shutil.copy(os.path.join(data_path, choice, file), os.path.join(save_path, 'all_' + choice, file))

    path_ims = os.path.join(save_path, 'all_images')
    path_masks = os.path.join(save_path, 'all_mask')
    path_gts = os.path.join(save_path, 'all_1st_labels_ah')

    all_im_names = sorted(os.listdir(path_ims))
    all_mask_names = sorted(os.listdir(path_masks))
    all_gt_names = sorted(os.listdir(path_gts))

    if len(all_im_names) != len(all_mask_names) or len(all_im_names) != len(all_gt_names):
        raise ValueError('Number of images, masks and ground truths do not match')

    all_im_names = [os.path.join(path_ims, n) for n in all_im_names]
    all_mask_names = [os.path.join(path_masks, n) for n in all_mask_names]
    all_gt_names = [os.path.join(path_gts, n) for n in all_gt_names]

    train_im_names = all_im_names[:16]
    test_im_names = all_im_names[16:]
    train_mask_names = all_mask_names[:16]

    test_mask_names = all_mask_names[16:]
    train_gt_names = all_gt_names[:16]
    test_gt_names = all_gt_names[16:]

    df_stare_all = pd.DataFrame({'im_paths': all_im_names,
                                    'gt_paths': all_gt_names,
                                    'mask_paths': all_mask_names})

    df_stare_train = pd.DataFrame({'im_paths': train_im_names,
                                    'gt_paths': train_gt_names,
                                    'mask_paths': train_mask_names})

    df_stare_test = pd.DataFrame({'im_paths': test_im_names,
                                    'gt_paths': test_gt_names,
                                    'mask_paths': test_mask_names})

    num_ims = len(df_stare_train)
    tr_ims = int(0.8 * num_ims)
    tr_ims = 12
    df_stare_train, df_stare_val = df_stare_train[:tr_ims], df_stare_train[tr_ims:]

    df_stare_train.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    df_stare_val.to_csv(os.path.join(save_path, 'val.csv'), index=False)
    df_stare_test.to_csv(os.path.join(save_path, 'test.csv'), index=False)
    df_stare_all.to_csv(os.path.join(save_path, 'test_all.csv'), index=False)
    print('STARE data prepared')
    return

def process_HRF(data_path, save_path):
    # 20train 5val 20test
    print('preparing HRF data')
    # process drive HRF, generate CSVs
    data_path = os.path.join(data_path, 'HRF')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"file {data_path} does not exists.")
    save_path = os.path.join(save_path, 'HRF')
    os.makedirs(save_path, exist_ok=True)

    choice_arr = ['images', 'manual1', 'mask']
    for choice in choice_arr:
        dir_path = os.path.join(save_path, 'all_' + choice)
        os.makedirs(dir_path, exist_ok=True)
        files = os.listdir(os.path.join(data_path, choice))
        for file in files:
            shutil.copy(os.path.join(data_path, choice, file), os.path.join(save_path, 'all_' + choice, file))

    path_ims = os.path.join(save_path, 'all_images')
    path_masks = os.path.join(save_path, 'all_mask')
    path_gts = os.path.join(save_path, 'all_manual1')

    path_ims_resized = os.path.join(save_path, 'images_resized')
    os.makedirs(path_ims_resized, exist_ok=True)
    path_masks_resized = os.path.join(save_path, 'mask_resized')
    os.makedirs(path_masks_resized, exist_ok=True)
    path_gts_resized = os.path.join(save_path, 'manual1_resized')
    os.makedirs(path_gts_resized, exist_ok=True)

    all_im_names = sorted(os.listdir(path_ims))
    all_mask_names = sorted(os.listdir(path_masks))
    all_gt_names = sorted(os.listdir(path_gts))

    if len(all_im_names) != len(all_mask_names) or len(all_im_names) != len(all_gt_names):
        raise ValueError('Number of images, masks and ground truths do not match')

    all_im_names = [os.path.join(path_ims, n) for n in all_im_names]
    all_mask_names = [os.path.join(path_masks, n) for n in all_mask_names]
    all_gt_names = [os.path.join(path_gts, n) for n in all_gt_names]

    df_hrf_all = pd.DataFrame({'im_paths': all_im_names,
                               'gt_paths': all_gt_names,
                               'mask_paths': all_mask_names})

    train_im_names = all_im_names[:25]
    test_im_names = all_im_names[25:]

    train_mask_names = all_mask_names[:25]
    test_mask_names = all_mask_names[25:]

    train_gt_names = all_gt_names[:25]
    test_gt_names = all_gt_names[25:]

    train_im_names_resized = [n.replace(path_ims, path_ims_resized) for n in train_im_names]
    train_mask_names_resized = [n.replace(path_masks, path_masks_resized) for n in train_mask_names]
    train_gt_names_resized = [n.replace(path_gts, path_gts_resized) for n in train_gt_names]

    df_hrf_train = pd.DataFrame({'im_paths': train_im_names_resized,
                                'gt_paths': train_gt_names_resized,
                                'mask_paths': train_mask_names_resized})

    df_hrf_test = pd.DataFrame({'im_paths': test_im_names,
                                'gt_paths': test_gt_names,
                                'mask_paths': test_mask_names})

    num_ims = len(df_hrf_train)
    tr_ims = int(0.8 * num_ims)
    tr_ims = 20
    df_hrf_train, df_hrf_val = df_hrf_train[:tr_ims], df_hrf_train[tr_ims:]

    df_hrf_train.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    df_hrf_val.to_csv(os.path.join(save_path, 'val.csv'), index=False)
    df_hrf_test.to_csv(os.path.join(save_path, 'test.csv'), index=False)
    df_hrf_all.to_csv(os.path.join(save_path, 'test_all.csv'), index=False)

    df_hrf_train_full_res = pd.DataFrame({'im_paths': train_im_names,
                                        'gt_paths': train_gt_names,
                                        'mask_paths': train_mask_names})
    df_hrf_train_full_res, df_hrf_val_full_res = df_hrf_train_full_res[:tr_ims], df_hrf_train_full_res[tr_ims:]
    df_hrf_train_full_res.to_csv(os.path.join(save_path, 'train_full_res.csv'), index=False)
    df_hrf_val_full_res.to_csv(os.path.join(save_path, 'val_full_res.csv'), index=False)

    print('Resizing HRF images ')

    for i in range(len(all_im_names)):
        im_name = all_im_names[i]
        im_name_out = im_name.replace('/all_images/', '/images_resized/').replace('\\all_images\\', '\\images_resized\\')
        im = Image.open(im_name)
        im_res = resize(im, size=(im.size[1] // 2, im.size[0] // 2), interpolation=Image.BICUBIC)
        im_res.save(im_name_out)

        mask_name = (im_name.replace('/all_images/', '/all_mask/').replace('.JPG', '_mask.tif')
                     .replace('.jpg', '_mask.tif').replace('\\all_images\\', '\\all_mask\\'))
        mask_name_out = mask_name.replace('/all_mask/', '/mask_resized/').replace('\\all_mask\\', '\\mask_resized\\')
        mask = Image.open(mask_name)
        mask_res = resize(mask, size=(mask.size[1] // 2, mask.size[0] // 2), interpolation=Image.NEAREST)
        # get rid of three channels in mask
        mask = Image.fromarray(np.array(mask)[:, :, 0])
        mask_res = Image.fromarray(np.array(mask_res)[:, :, 0])
        mask.save(mask_name)
        mask_res.save(mask_name_out)

        gt_name = (im_name.replace('/all_images/', '/all_manual1/').replace('.JPG', '.tif')
                   .replace('.jpg', '.tif').replace('\\all_images\\', '\\all_manual1\\'))
        gt_name_out = gt_name.replace('/all_manual1/', '/manual1_resized/').replace('\\all_manual1\\', '\\manual1_resized\\')
        gt = Image.open(gt_name)
        gt_res = resize(gt, size=(gt.size[1] // 2, gt.size[0] // 2), interpolation=Image.NEAREST)
        gt_res.save(gt_name_out)
    print('HRF data prepared')
    return

if __name__ == '__main__':
    data_path = r"/olddata/" # olddata path
    save_path = r"/data/"    # savedata path
    process_DRIVE(data_path, save_path)
    process_CHASEDB1(data_path, save_path)
    process_STARE(data_path, save_path)
    process_HRF(data_path, save_path)