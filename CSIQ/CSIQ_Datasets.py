import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms

from util.data_process import crop_patches


"""
一定要用附属的xlsx文件，其实就是源文件的第一个sheet，以第一个sheet的方式组织数据
训练用的失真登记为1-4个数据，第五个等级作为测试数据，第一版训练和测试数据安排的不怎么合理，可以自己改
"""
excel_file_path = './data/csiq-dmos.xlsx'
df = pd.read_excel(excel_file_path, sheet_name='all_by_image')

#失真图种类
def get_dst_types():
    df_unique = df.drop_duplicates(subset='dst_type')
    dst_types = df_unique['dst_type']
    dst_types_list = dst_types.tolist()
    # 使用循环和 remove() 方法剔除'noise'
    while 'noise' in dst_types_list:
        dst_types_list.remove('noise')
        dst_types_list.append('fnoise')
    return dst_types_list

#参考图类别名称
def get_ref_names():
    df_unique = df.drop_duplicates(subset='image')
    ref_names = df_unique['image']
    ref_name_list = ref_names.tolist()

    return ref_name_list

#失真图文件名称和dmos分数
def get_dst_filename_dmos(head_x, is_train = True):
    if is_train:
        filtered_df = df[(df['image'] == head_x) & (df['dst_idx'] != 1) & (df['dst_lev'] != 5)]
    else:
        filtered_df = df[(df['image'] == head_x) & (df['dst_idx'] != 1) & (df['dst_lev'] == 5)]

    selected_columns = filtered_df[['image', 'dst_type', 'dst_lev', 'dmos']]
    dst_filename_dmos_list = selected_columns.values.tolist()
    filename_list = []
    dmos_list = []
    for data in dst_filename_dmos_list:
        dst_type = str(data[1]).strip()
        if dst_type == 'awgn' or dst_type == 'blur' or dst_type == 'jpeg':
            dst_type = dst_type.upper()
        filename_list.append(str(data[0]).strip() + '.' + dst_type + '.' + str(data[2]).strip() + '.png')
        dmos_list.append(data[3])
    return filename_list, dmos_list

class PairedDataset(Dataset):
    def __init__(self, ref_dir, distorted_dir, image_name, is_train=True, transform=None):
        """
        ref_dir: 参考图片的目录路径
        distorted_dir: 失真图片的目录路径
        transform: 预处理和数据的转换
        """
        self.ref_dir = ref_dir
        self.distorted_dir = distorted_dir
        self.transform = transform
        self.image_name = image_name
        # self.dst_filenames = [f for f in os.listdir(self.distorted_dir) if str(self.image_name) in f and '5' not in f]
        self.dst_filenames, self.dst_dmos = get_dst_filename_dmos(self.image_name, is_train)
        pass

    def __len__(self):
        return len(self.dst_filenames)

    def __getitem__(self, idx):
        ref_path = os.path.join(self.ref_dir, str(self.image_name) + '.png')
        distorted_path = os.path.join(self.distorted_dir, self.dst_filenames[idx])

        ref_image = Image.open(ref_path).convert('RGB')
        distorted_image = Image.open(distorted_path).convert('RGB')

        data = crop_patches(distorted_image, ref_image, is_overlap=True)
        dist_patches = data[0].unsqueeze(0)
        ref_patches = data[1].unsqueeze(0)

        dmos_score = self.dst_dmos[idx]

        return ref_patches, dist_patches, dmos_score



if __name__ == '__main__':
# 使用示例
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    ref_dir = './data/src_imgs'
    distorted_dir = './data/all_dst_imgs'
    ref_name_list = get_ref_names() #参考图像文件名
    for epoch in range(5):
        for ref_name in ref_name_list:
            dataset = PairedDataset(ref_dir, distorted_dir, ref_name, True, transform=transform)
            dataloader = DataLoader(dataset, batch_size=6, shuffle=True)
            for data in dataloader:
                ref_temp, distorted_temp, dmos_score = data



