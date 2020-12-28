# -*- coding=utf-8 -*-
import cv2
import os
import shutil


def select(img_dir_1, img_dir_2, cp_source, cp_des):
    '''
    比较两个文件夹中的图像
    '''
    case_names_1 = os.listdir(img_dir_1)
    file_name_1 = []
    for case_name in case_names_1:
        if '.png' in case_name:
            file_name_1.append(case_name.split('.')[0])

    print(len(file_name_1))

    case_names_2 = os.listdir(img_dir_2)
    file_name_2 = []
    diff_file_name_2 = []
    for case_name in case_names_2:
        file_name_2.append(case_name.split('.')[0])


    print(len(file_name_2))

    for file in file_name_1:
        if file not in file_name_2:
            diff_file_name_2.append(file)

    print(len(diff_file_name_2))
    print(diff_file_name_2)

    for name in diff_file_name_2:
        # source_name = os.path.join(cp_source,name+'.png')
        # shutil.copy(source_name, cp_des)
        os.remove(os.path.join(cp_source,name+'.png'))
        os.remove(os.path.join(cp_des,name+'.npy'))
        os.remove(os.path.join(cp_des,name+'.png'))




if __name__ == '__main__':
    label = 'fold_3/3_high_grade/'
    # select('/data/hdd1/syh/PycharmProjects/CGC-Net/data_final_add/proto/mask/CRC/'+label,'/data/hdd1/syh/PycharmProjects/CGC-Net/data_final_add/proto/coordinate/CRC/'+label,
           # '/data/hdd1/syh/PycharmProjects/CGC-Net/data_final_add/raw/CRC/'+label,'/data/hdd1/syh/PycharmProjects/CGC-Net/data_final_add/proto/mask/CRC/' + label)
    select('/data/hdd1/syh/PycharmProjects/CGC-Net/data_final/proto/mask/CRC/' + label, '/data/hdd1/syh/PycharmProjects/CGC-Net/data_final/proto/coordinate/CRC/' + label,
           '/data/hdd1/syh/PycharmProjects/CGC-Net/data_final/raw/CRC/'+label,'/data/hdd1/syh/PycharmProjects/CGC-Net/data_final/proto/mask/CRC/' + label)