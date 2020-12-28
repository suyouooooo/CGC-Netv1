import numpy as np
import os
import sys
sys.path.append('../')
from skimage.measure import regionprops
from skimage.filters.rank import entropy as Entropy
from skimage.morphology import disk,remove_small_objects
from common.utils import mkdirs
from common.nuc_feature import nuc_stats_new,nuc_glcm_stats_new
import multiprocessing
import time
import cv2
import shutil


# H, W =3584,3584
H, W = 1792, 1792

def euc_dist(name):
    arr = np.load(name)
    arr_x = (arr[:,0,np.newaxis].T - arr[:,0,np.newaxis])**2
    arr_y = (arr[:,1,np.newaxis].T - arr[:,1,np.newaxis])**2
    arr = np.sqrt(arr_x + arr_y)

    np.save(name.replace('coordinate', 'distance'), arr.astype(np.int16))
    return 0

class DataSetting:
    def __init__(self, label = 'fold_3/2_low_grade'):
        self.dataset = 'CRC'
        self.label  = label
        self.root = '/data/hdd1/syh/PycharmProjects/CGC-Net/data_final_add/raw'#'/research/dept6/ynzhou/gcnn/data/raw'
        self.test_data_path = os.path.join(self.root, self.dataset, self.label)
        self.save_path = '/data/hdd1/syh/PycharmProjects/CGC-Net/data_final_add/proto'#'/research/dept6/ynzhou/gcnn/data/proto'
        self.do_eval = False
        self.test_image_list = os.listdir(self.test_data_path)
        self.test_image_list = [f for f in self.test_image_list if 'png' in f]
        self.test_data = self.test_image_list
        self.des_path_mask = '/data/hdd1/syh/PycharmProjects/CGC-Net/data_final_add/all_dark'
        self.des_path_mask_png = '/data/hdd1/syh/PycharmProjects/CGC-Net/data_final_add/all_dark_png'
        self.des_path_raw = '/data/hdd1/syh/PycharmProjects/CGC-Net/data_final_add/all_raw'


class GraphSetting(DataSetting):
    def __init__(self):
        super(GraphSetting, self).__init__()
        self.numpy_list = os.listdir(os.path.join(self.save_path, 'mask',self.dataset, self.label))
        self.numpy_list = [ f for f in self.numpy_list if 'npy' in f]
        self.feature_save_path = os.path.join(self.save_path,'feature', self.dataset, self.label, )
        self.distance_save_path = os.path.join(self.save_path,'coordinate', self.dataset, self.label)
        mkdirs(self.distance_save_path)
        mkdirs(self.feature_save_path)
        mkdirs(self.distance_save_path.replace('coordinate', 'distance'))

def _get_batch_features_new(numpy_queue, info_queue):
    self = GraphSetting()
    while True:
        name = numpy_queue.get()
        if name != 'end':
            # prepare original image
            mask_npy = os.path.join(self.save_path, 'mask', self.dataset, self.label, name)
            raw_png = os.path.join(self.test_data_path, name.replace('.npy', '.png'))
            mask = np.load(mask_npy)
            des_mask_path = os.path.join(self.des_path_mask, self.dataset, self.label)
            des_mask_path_png = os.path.join(self.des_path_mask_png, self.dataset, self.label)
            des_raw_path = os.path.join(self.des_path_raw, self.dataset, self.label)
            mask_png = mask_npy.replace('.npy', '.png')
            # mask = np.load(os.path.join(self.save_path, 'mask',self.dataset, self.label, name))
            # print('mask.shape' + str(mask.shape))
            # print('type(mask.shape)' + str(type(mask.shape)))
            H,W = mask.shape[0], mask.shape[1]
            mask = remove_small_objects(mask, min_size=10, connectivity=1, in_place=True)
            ori_image = cv2.imread(os.path.join(self.test_data_path, name.replace('.npy', '.png')))
            # print("ori_image shape " + str(ori_image.shape))
            # cv2.COLOR_BGR2GRAY 会让图片变黄，将三维图片转换为二维
            int_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
            # print("int_image1 shape " + str(int_image.shape))
            int_image = cv2.resize(int_image, (W,H), interpolation=cv2.INTER_LINEAR)
            # print("int_image2 shape " + str(int_image.shape))
            entropy = Entropy(int_image, disk(3))
            # print("int_image3 shape " + str(int_image.shape))
            # regionprops用来测量标记图像区域的属性，比如连通域的面积，外接矩形的面积，连通域的质心等等
            props = regionprops(mask)
            # 将mask变成一个二值图像
            binary_mask = mask.copy()
            binary_mask[binary_mask>0]= 1
            node_feature = []
            node_coordinate = []
            for prop in props:
                nuc_feats = []
                bbox = prop.bbox
                single_entropy = entropy[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
                single_mask = binary_mask[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1].astype(np.uint8)

                single_int = int_image[bbox[0]: bbox[2]+1, bbox[1]:bbox[3]+1]
                # 提取质心特征，centroid表示质心坐标
                coor = prop.centroid
                # single_mask代指mask图像，single_int代指原始的图像，根据两种图像的不同提取出细胞核的16种特征
                mean_im_out, diff, var_im, skew_im = nuc_stats_new(single_mask,single_int)
                glcm_feat = nuc_glcm_stats_new(single_mask, single_int) # just breakline for better code
                glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_ASM = glcm_feat
                mean_ent = cv2.mean(single_entropy, mask=single_mask)[0]
                info = cv2.findContours(single_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnt = info[0][0]
                num_vertices = len(cnt)
                area = cv2.contourArea(cnt)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    hull_area += 1
                solidity = float(area)/hull_area
                if num_vertices > 4:
                    centre, axes, orientation = cv2.fitEllipse(cnt)
                    majoraxis_length = max(axes)
                    minoraxis_length = min(axes)
                else:
                    orientation = 0
                    majoraxis_length = 1
                    minoraxis_length = 1
                perimeter = cv2.arcLength(cnt, True)
                eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)
                nuc_feats.append(mean_im_out)
                nuc_feats.append(diff)
                nuc_feats.append(var_im)
                nuc_feats.append(skew_im)
                nuc_feats.append(mean_ent)
                nuc_feats.append(glcm_dissimilarity)
                nuc_feats.append(glcm_homogeneity)
                nuc_feats.append(glcm_energy)
                nuc_feats.append(glcm_ASM)
                nuc_feats.append(eccentricity)
                nuc_feats.append(area)
                nuc_feats.append(majoraxis_length)
                nuc_feats.append(minoraxis_length)
                nuc_feats.append(perimeter)
                nuc_feats.append(solidity)
                nuc_feats.append(orientation)
                feature = np.hstack(nuc_feats)
                node_feature.append(feature)
                node_coordinate.append(coor)

            # print("len(node_feature) : " + str(len(node_feature)))
            if len(node_feature) == 0:
                ## TODO 会出现错误ValueError: need at least one array to concatenate
                # print("0 name :" + name)
                # all_dark_list.append(name)
                shutil.move(mask_npy, des_mask_path)
                shutil.move(mask_png, des_mask_path_png)
                shutil.move(raw_png, des_raw_path)
            else:
                node_feature = np.vstack(node_feature)
                node_coordinate = np.vstack(node_coordinate)

                # 把特征表示和质心表示都存储起来
                np.save(os.path.join(self.feature_save_path, name), node_feature.astype(np.float32))
                np.save(os.path.join(self.distance_save_path, name), node_coordinate.astype(np.float32))

                # 计算质心之间的euc_dist
                euc_dist(os.path.join(self.distance_save_path, name))


                info_queue.put(name)
        else:
            break

if __name__ == '__main__':
    setting = GraphSetting()
    nameQueue = multiprocessing.Queue()
    infoQueue = multiprocessing.Queue()
    # all_dark_list = []

    for i in setting.numpy_list:
        nameQueue.put(i)
    nameQueue.put('end')
    Process_C = []
    for i in range(10):
        Process_C.append(multiprocessing.Process(target=_get_batch_features_new, args=(nameQueue, infoQueue)))
    for i in range(10):
        Process_C[i].start()

    total = len(setting.numpy_list)
    finished = 0
    while True:
        token_info = nameQueue.empty()
        finish_name = infoQueue.get(1)
        if finish_name in setting.numpy_list:
            finished += 1
        if finished % 5 == 0:
            print('Finish %d/%d'%(finished, total))
        if token_info:
            print('finished')
            time.sleep(10)
            for i in range(10):
                Process_C[i].terminate()
            for i in range(10):
                Process_C[i].join()
            break
    # print("all_dark_list: " + str(all_dark_list))
