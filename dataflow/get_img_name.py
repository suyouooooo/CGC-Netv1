# -*- coding=utf-8 -*-
import cv2
import os
import shutil
import time


def select(img_dir_1):
    '''
    比较两个文件夹中的图像
    '''
    case_names_1 = os.listdir(img_dir_1)
    file_name_1 = []
    for case_name in case_names_1:
        if '.png' in case_name:
            if 'Normal' in case_name:
                case_name = case_name.replace('Normal', 'grade_1')
            if 'Low-Grade' in case_name:
                case_name = case_name.replace('Low-Grade', 'grade_2')
            if 'High-Grade' in case_name:
                case_name = case_name.replace('High-Grade', 'grade_3')
            file_name_1.append(case_name.split('.')[0])


    print(len(file_name_1))

    print(file_name_1)

    # case_names_2 = os.listdir(img_dir_2)
    # file_name_2 = []
    # diff_file_name_2 = []
    # for case_name in case_names_2:
    #     file_name_2.append(case_name.split('.')[0])
    #
    #
    # print(len(file_name_2))
    #
    # for file in file_name_1:
    #     if file not  in file_name_2:
    #         diff_file_name_2.append(file)
    #
    # print(len(diff_file_name_2))
    # print(diff_file_name_2)

    # if len(diff_file_name_2) != 0:


    # for name in diff_file_name_2:
    #     source_name = os.path.join(cp_source,name+'.png')
    #     shutil.copy(source_name, cp_des)
    #     source_name2 = os.path.join(cp_source, name + '.npy')
    #     shutil.copy(source_name2, cp_des)
    #     time.sleep(5)



if __name__ == '__main__':
    fold_list = ['fold_1/', 'fold_2/', 'fold_3/']
    # fold_list = [ 'fold_3/']
    grade_list = ['1_normal/', '2_low_grade/', '3_high_grade/']

    for fold in fold_list:
        for grade in grade_list:
            label = fold+grade
            print(label)
            select('/data/hdd1/syh/PycharmProjects/CGC-Net/data/raw/CRC/'+label)