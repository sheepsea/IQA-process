import os

import numpy as np
import torch.nn.functional as F
import scipy.io as sio
import torch
from PIL import Image
from scipy.signal import convolve2d
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

# LIVE文件夹存放路径
ref_dir = '../data/LIVE/'

mat_data = sio.loadmat(ref_dir + 'dmos.mat')
name_data = sio.loadmat(ref_dir + 'refnames_all.mat')

#获取参考图像的文件路径，文件名称，DMOS值
def get_ref_list(train_type='train'):
    dst_index_dict = {'jp2k': 227, 'jpeg': 233, 'wn': 174, 'gblur': 174, 'fastfading': 174}
    head = 0
    dst_name_path = []
    ref_list = []
    dmos_list = []
    ref_name_list = []

    list_from_array = name_data['refnames_all'].tolist()
    name_list = [item for sublist in list_from_array for item in sublist]
    for ref_name in name_list:
        ref_name_list.append(ref_name[0])

    dmos_all_list = mat_data['dmos'][0].tolist()

    for key, value in dst_index_dict.items():
        # 计算分割点
        first_split = int(0.80 * value)  # 前70%的数据量
        second_split = int(0.95 * value)  # 后15%开始的位置，即前70% + 15%
        last_split = value - second_split
        # 分割对应列表
        end = head + value
        ref_name_sublist = ref_name_list[head:end]
        dmos_all_sublist = dmos_all_list[head:end]
        head = end

        if train_type == 'train':
            ref_list = ref_list + ref_name_sublist[:first_split]
            dmos_list = dmos_list + dmos_all_sublist[:first_split]
            for i in range(first_split):
                dst_name_path.append(key + '/img' + str(i + 1) + '.bmp')
        elif train_type == 'val':
            ref_list = ref_list + ref_name_sublist[first_split:second_split]
            dmos_list = dmos_list + dmos_all_sublist[first_split:second_split]
            for i in range(second_split - first_split):
                dst_name_path.append(key + '/img' + str(first_split + i + 1) + '.bmp')
        elif train_type == 'test':
            ref_list = ref_list + ref_name_sublist[second_split:]
            dmos_list = dmos_list + dmos_all_sublist[second_split:]
            for i in range(last_split):
                dst_name_path.append(key + '/img' + str(second_split + i + 1) + '.bmp')
    return ref_list, dst_name_path, dmos_list

class LIVEDataset(Dataset):
    def __init__(self, train_type='train', transform=None):

        self.ref_dir = ref_dir
        self.transform = transform

        if train_type == 'train':
            self.ref_list, self.dst_name_path, self.dmos_list = get_ref_list(train_type)
        elif train_type == 'val':
            self.ref_list, self.dst_name_path, self.dmos_list = get_ref_list(train_type)
        elif train_type == 'test':
            self.ref_list, self.dst_name_path, self.dmos_list = get_ref_list(train_type)


    def __len__(self):
        return len(self.dst_name_path)

    def __getitem__(self, idx):
        ref_img = Image.open(self.ref_list[idx]).convert('RGB')
        dst_img = Image.open(self.dst_name_path[idx]).convert('RGB')

        return ref_img, dst_img, self.dmos_list[idx]







