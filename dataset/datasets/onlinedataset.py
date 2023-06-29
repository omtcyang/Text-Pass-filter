import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
from copy import deepcopy as cdc

import torch
from torch.utils import data
import torchvision.transforms as transforms

from .aug import *
from .supervision import supervision_generate
from .parser import load_img, parse_label

INF = 1e6


class OnlineDataset(data.Dataset):
    def __init__(self,
                 dataset_name,
                 data_path,
                 split='train',
                 short_size=None,
                 sample_strides=None,
                 sample_types=None,
                 mask_size=None,
                 is_ignore=False,
                 is_transform=False,
                 report_speed=False):

        self.dataset_name = dataset_name
        self.split = split
        self.short_size = short_size
        # 4 8 16 32
        if split == 'train':
            self.sample_stride = [int(k) for k, v in sample_strides.items() if v == 1]
            # single / multi
            self.sample_type = [k for k, v in sample_types.items() if v == 1][0]
            self.sample_scale_ranges = []
            self.init_scale_ranges()

        self.mask_size = mask_size
        self.is_ignore = is_ignore
        self.is_transform = is_transform

        self.img_paths, self.imgs, self.instances, self.ignores = [], [], [], []
        self.boxes, self.texts = [], []

        # 1. 从配置文件加载数据路径
        self.data_dir, self.gt_dir = data_path.img, data_path.gt

        # 2. 解析数据集标签信息
        labels = parse_label(dataset_name, split, self.data_dir, self.gt_dir)
        if len(labels) == 4:
            self.img_paths, self.imgs, self.instances, self.ignores = labels
        elif len(labels) == 3:
            self.img_paths, self.boxes, self.texts = labels

        # 3. 配置打印模型速度相关参数
        if report_speed:
            target_size = 3000
            data_size = len(self.img_paths)
            extend_scale = (target_size + data_size - 1) // data_size
            self.img_paths = (self.img_paths * extend_scale)[:target_size]
        self.max_word_num = 200

    def __len__(self):
        return len(self.img_paths)

    def init_scale_ranges(self):
        if len(self.sample_stride) == 1:
            self.sample_scale_ranges.append([-1, INF])
            return
        last = -1
        for i in range(len(self.sample_stride) - 1):
            now = self.short_size / (2 ** (len(self.sample_stride) - i))
            self.sample_scale_ranges.append([last, now])
            last = now
        self.sample_scale_ranges.append([last, INF])

    def prepare_train_data(self, index):
        # 1. 当数据集为SynthText时
        if self.dataset_name == 'SynthText':
            img_path, gt_box, gt_text = self.img_paths[index], self.boxes[index], self.texts[index]
            img = np.array(Image.open(img_path))

            # 处理文本对应的text标签
            text = []
            for gtt in gt_text:
                gtt = gtt.split('\n')
                for gt in gtt:
                    gt = gt.strip().split(' ')
                    for t in gt:
                        text.append(t)

            # 处理文本对应的box标签     boxes: [2,4,n], n为word的个数
            gt_box = gt_box.astype(np.int32)
            if len(gt_box.shape) == 2:
                # 此时是2维矩阵，需要扩展一个维度，和3维矩阵格式保持一致
                gt_box = gt_box[:, :, None]

            # 画instance
            instance = np.zeros(img.shape[:2], np.uint8)
            for idx in range(gt_box.shape[2]):
                sub_box = [
                    [gt_box[0][0][idx], gt_box[1][0][idx]],
                    [gt_box[0][1][idx], gt_box[1][1][idx]],
                    [gt_box[0][2][idx], gt_box[1][2][idx]],
                    [gt_box[0][3][idx], gt_box[1][3][idx]]
                ]
                sub_box = np.int32(np.array(sub_box))[None, :, :]
                cv2.drawContours(instance, sub_box, 0, idx + 1, -1)

        # 2. 当数据集不是SynthText时
        else:
            img, instance = self.imgs[index], self.instances[index]  # img, instance

        # 3. 加载ignore
        if self.is_ignore:
            ignore = 1 - self.ignores[index]  # ignore
        else:
            ignore = np.ones(instance.shape)

        # 4. 数据增强
        if self.is_transform:
            # 进行几何方面的数据增强
            datas = [img, ignore, instance]
            datas = random_scale(datas, self.short_size)
            datas = random_horizontal_flip(datas)
            datas = random_rotate(datas)
            datas = random_crop_padding(datas, self.short_size)
            datas = supervision_generate(datas,
                                         self.img_paths[index],
                                         self.sample_stride,
                                         self.sample_type,
                                         self.sample_scale_ranges,
                                         self.mask_size)
            img, ignore, instance, center, centerness, dminimum = datas[0], \
                                                                  datas[1], \
                                                                  datas[2], \
                                                                  datas[3], \
                                                                  datas[4], \
                                                                  datas[5]

            # 单独对img进行颜色方面的数据增强
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        # 5. 图像归一化
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        # 6. 返回数据
        ignore = torch.from_numpy(ignore).long()

        ins = np.zeros((instance.shape[0] // 4, instance.shape[1] // 4), np.uint8)
        for u in np.unique(instance):
            if u == 0:
                continue
            u_temp = cv2.resize((instance == u) * 1.0, dsize=(instance.shape[1] // 4, instance.shape[0] // 4))
            contours, _ = cv2.findContours(u_temp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                cv2.drawContours(ins, [contour[:, 0, :]], 0, u, -1)
        # instance = cv2.resize(instance, dsize=(instance.shape[1] // 4, instance.shape[0] // 4))
        # instance = torch.from_numpy(instance).long()

        data = dict(
            imgs=img,
            gt_instance=ins,
            gt_ignore=ignore,
            gt_center=center,
            gt_centerness=centerness,
            gt_dminimum=dminimum,
            # imgs_meta=self.img_paths[index]
        )
        return data

    def prepare_test_data(self, index):
        # 1. 数据路径
        img_path = self.img_paths[index]
        img_meta = dict(
            img_path=img_path
        )

        # 2. 加载原尺寸图像
        img = load_img(img_path)
        img_ori = cdc(img)
        img_meta.update(dict(
            ori_size=np.array(img_ori.shape[:2])
        ))

        # 3. rescale图像
        img = scale_aligned_short(img, self.short_size)
        img_meta.update(dict(
            resize_size=np.array(img.shape[:2])
        ))

        # 4. 图像归一化
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        data = dict(
            imgs=img,
            gt_instance=self.instances[index],
            gt_ignore=self.ignores[index],
            imgs_ori=img_ori,
            imgs_meta=img_meta,
        )
        return data

    def __getitem__(self, index):
        if self.split == 'train':
            return self.prepare_train_data(index)
        elif self.split == 'test':
            return self.prepare_test_data(index)
