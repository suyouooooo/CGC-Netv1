# -*- coding=utf-8 -*-
import cv2
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from tqdm import tqdm
from PIL import Image
W, H = 3584,3584

def convertPNG2NPY(img_dir, save_dir):
    '''
    将PNG转成npy格式的数据
    '''
    case_names = os.listdir(img_dir)
    for case_name in tqdm(case_names):
        if 'png' in case_name:
            print(case_name)
            img_path = os.path.join(img_dir, case_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # resize images
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
            # print('mask png shape ' + str(image.shape))
            # print('mask npy shape ' + str(np.asarray(image, np.uint8).shape))
            np.save(os.path.join(save_dir,  "{}.npy".format(case_name[0:-len('.png')])), np.asarray(image, np.uint8))

            # 使照片发黄
            # int_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # int_image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
            # plt.imsave(os.path.join(save_dir, "{}.png".format(case_name[0:-len('.png')])), int_image,
            #            cmap='plasma')  # 定义命名规则，保存图片


def convertNPY2PNG(npy_dir, save_dir):
    case_names = os.listdir(npy_dir)
    for case_name in tqdm(case_names):
        print(case_name)
        npy_path = os.path.join(npy_dir, case_name)
        arr = np.load(npy_path)
        disp_to_img = np.array(Image.fromarray(arr).resize([3584,3584]))
        # disp_to_img = scipy.misc.imresize(arr, [3584,3584])  # 根据 需要的尺寸进行修改
        plt.imsave(os.path.join(save_dir, "{}.png".format(case_name[0:-len('.npy')])), disp_to_img,
                   cmap='plasma')  # 定义命名规则，保存图片


# convertPNG2NPY('images', 'images3')

# convertNPY2PNG('npys', 'images_after')
if __name__ == '__main__':
    # convertPNG2NPY(sys.argv[1], sys.argv[2])
    fold_list = ['fold_1/', 'fold_2/', 'fold_3/']
    grade_list = ['1_normal/', '2_low_grade/', '3_high_grade/']
    img_dir = 'data_final/proto/mask/CRC/'
    for fold in fold_list:
        for grade in grade_list:
            img_path = os.path.join(img_dir, fold, grade)
            print(img_path)
            convertPNG2NPY(img_path, img_path)
    # convertPNG2NPY('data/proto/mask/CRC/fold_1/1_normal', 'data/proto/mask/CRC/fold_1/1_normal')
    # convertPNG2NPY('data/proto/mask/CRC/fold_1/2_low_grade', 'data/proto/mask/CRC/fold_1/2_low_grade')
    # convertPNG2NPY('data/proto/mask/CRC/fold_1/3_high_grade', 'data/proto/mask/CRC/fold_1/3_high_grade')
    # convertPNG2NPY('data/proto/mask/CRC/fold_2/1_normal', 'data/proto/mask/CRC/fold_2/1_normal')
    # convertPNG2NPY('data/proto/mask/CRC/fold_2/2_low_grade', 'data/proto/mask/CRC/fold_2/2_low_grade')
    # convertPNG2NPY('data/proto/mask/CRC/fold_2/3_high_grade', 'data/proto/mask/CRC/fold_2/3_high_grade')
    # convertPNG2NPY('data/proto/mask/CRC/fold_3/1_normal', 'data/proto/mask/CRC/fold_3/1_normal')
    # convertPNG2NPY('data/proto/mask/CRC/fold_3/2_low_grade', 'data/proto/mask/CRC/fold_3/2_low_grade')
    # convertPNG2NPY('data/proto/mask/CRC/fold_3/3_high_grade', 'data/proto/mask/CRC/fold_3/3_high_grade')