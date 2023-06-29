import os
import os.path as osp

import numpy as np
from PIL import Image
import cv2
import math
import scipy.io as scio


def load_img(img_path):
    try:
        img = Image.open(img_path)
        img_shape = np.array(img).shape
        if len(img_shape) == 2:
            img = img.convert('RGB')
        if (len(img_shape) == 3) and (img_shape[2] == 4):
            img = img.convert('RGB')
        img = np.array(img)
    except Exception as e:
        raise ValueError('Failed loading img:', img_path)

    return img


def rotate(angle, x, y):
    """
    基于原点的弧度旋转
    :param angle:   弧度
    :param x:       x
    :param y:       y
    :return:
    """
    rotatex = math.cos(angle) * x - math.sin(angle) * y
    rotatey = math.cos(angle) * y + math.sin(angle) * x
    return int(rotatex), int(rotatey)


def xy_rorate(theta, x, y, centerx, centery):
    """
    针对中心点进行旋转
    :param theta:
    :param x:
    :param y:
    :param centerx:
    :param centery:
    :return:
    """
    r_x, r_y = rotate(theta, x - centerx, y - centery)
    return centerx + r_x, centery + r_y


def rec_rotate(x, y, width, height, theta):
    """
    传入矩形的x,y和宽度高度，弧度，转成QUAD格式
    :param x:
    :param y:
    :param width:
    :param height:
    :param theta:
    :return:
    """
    centerx = int(x + width / 2)
    centery = int(y + height / 2)

    x1, y1 = xy_rorate(theta, x, y, centerx, centery)
    x2, y2 = xy_rorate(theta, x + width, y, centerx, centery)
    x3, y3 = xy_rorate(theta, x, y + height, centerx, centery)
    x4, y4 = xy_rorate(theta, x + width, y + height, centerx, centery)

    return x1, y1, x2, y2, x3, y3, x4, y4


def parse_MSRA(dataset_name, split, data_dir, gt_dir):
    img_paths, imgs, instances, ignores = [], [], [], []

    for num, gt_file in enumerate(os.listdir(gt_dir)):

        # GT 路径
        if split == 'test':
            img_file = str(gt_file.split('.')[0]) + '.png'
        else:
            img_file = str(gt_file.split('.')[0]) + '.jpg'
        img_path = osp.join(data_dir, img_file)
        img = load_img(img_path)

        # 初始化 instance_mask and ignore_mask 
        instance = np.zeros(img.shape[:2], np.uint8)
        ignore = np.zeros(img.shape[:2], np.uint8)

        gt_path = osp.join(gt_dir, gt_file)
        with open(gt_path, 'r', encoding='utf-8') as f:
            content = f.readlines()

        for idx, cont in enumerate(content):
            cont = cont.replace('\n', '').split(' ')
            x, y, w, h, theta = eval(cont[2]), eval(cont[3]), eval(cont[4]), eval(cont[5]), eval(cont[6])
            x1, y1, x2, y2, x3, y3, x4, y4 = rec_rotate(x, y, w, h, theta)
            points = [[x1, y1], [x2, y2], [x4, y4], [x3, y3]]
            points = np.array(points)[None, :, :]

            if eval(cont[1]) != 0:
                cv2.drawContours(ignore, points, 0, 1, -1)
                continue  # ignore
            cv2.drawContours(instance, points, 0, idx + 1, -1)  # instance

        img_paths.append(img_path)
        imgs.append(img)
        instances.append(instance)
        ignores.append(ignore)

    return [img_paths, imgs, instances, ignores]


def parse_TT(dataset_name, split, data_dir, gt_dir):
    img_paths, imgs, instances, ignores = [], [], [], []

    for num, gt_file in enumerate(os.listdir(gt_dir)):

        # GT 路径
        img_file = str(gt_file.split('.')[0].split('_')[-1]) + '.jpg'
        img_path = osp.join(data_dir, img_file)
        img = load_img(img_path)

        # 初始化 instance_mask and ignore_mask 
        instance = np.zeros(img.shape[:2], np.uint8)
        ignore = np.zeros(img.shape[:2], np.uint8)

        # 遍历当前gt文件中的信息, 生成 instance, ignore
        gt_path = osp.join(gt_dir, gt_file)
        gt_data = scio.loadmat(gt_path)['polygt']
        for idx, cell in enumerate(gt_data):
            coorX, coorY = cell[1][0], cell[3][0]
            if len(coorX) < 4: continue  # too few points
            points = np.stack([coorX, coorY]).T.astype(np.int32)[None, :, :]  # points : n*2

            strTrans = cell[4][0] if len(cell[4]) > 0 else '#'
            if strTrans == '#': cv2.drawContours(ignore, points, 0, 1, -1); continue  # ignore instance
            cv2.drawContours(instance, points, 0, idx + 1, -1)

        img_paths.append(img_path)
        imgs.append(img)
        instances.append(instance)
        ignores.append(ignore)

    return [img_paths, imgs, instances, ignores]


def parse_CTW(dataset_name, split, data_dir, gt_dir):
    img_paths, imgs, instances, ignores = [], [], [], []

    for num, gt_file in enumerate(os.listdir(gt_dir)):

        # GT 路径
        img_file = str(gt_file.split('.')[0]) + '.jpg'
        img_path = osp.join(data_dir, img_file)
        img = load_img(img_path)

        # 初始化 instance_mask 
        instance = np.zeros(img.shape[:2], np.uint8)

        # 遍历当前gt文件中的信息, 生成 instance, ignore
        gt_path = osp.join(gt_dir, gt_file)
        with open(gt_path, 'r', encoding='utf-8') as f:
            content = f.readlines()
        for idx, cont in enumerate(content):
            cont = cont.replace('\n', '').split(',')

            cont_bbox = cont[:4]
            bbox = []
            for i in range(0, len(cont_bbox), 2):
                bbox.append([eval(cont_bbox[i]), eval(cont_bbox[i + 1])])

            cont_curve = cont[4:]
            points = []
            for i in range(0, len(cont_curve), 2):
                points.append([eval(cont_curve[i]) + bbox[0][0], eval(cont_curve[i + 1]) + bbox[0][1]])

            points = np.array(points)[None, :, :]
            cv2.drawContours(instance, points, 0, idx + 1, -1)

        img_paths.append(img_path)
        imgs.append(img)
        instances.append(instance)

    return [img_paths, imgs, instances, ignores]


def parse_IC15(dataset_name, split, data_dir, gt_dir):
    img_paths, imgs, instances, ignores = [], [], [], []

    for num, img_file, gt_file in enumerate(zip(data_dir, gt_dir)):

        # GT 路径
        img_path = os.path.join(data_dir, img_file)
        img = load_img(img_path)

        # 初始化 instance_mask 
        instance = np.zeros(img.shape[:2], np.uint8)
        ignore = np.zeros(img.shape[:2], np.uint8)

        # 遍历当前gt文件中的信息, 生成 instance, ignore
        gt_path = os.path.join(gt_dir, gt_file)
        with open(gt_path, 'r', encoding='utf-8') as f:
            content = f.readlines()
            # train.GT 文件 每行的最开始有一个\ufeff377，
            # 如['\ufeff377,117,463,117,465,130,378,130,Genaxis Theatre\n']
            if self.split == 'train':
                content[0] = content[0][1:]
            for idx, cont in enumerate(content):
                cont = cont.split(',')

                # x1,y1 ... , x4,y4
                points = [[cont[0], cont[1]], [cont[2], cont[3]], [cont[4], cont[5]], [cont[6], cont[7]]]
                points = np.array(points)[None, :, :]

                # 训练集中不画出 ### 文本的mask，当做全黑负样本处理
                if cont[-1][:-1] == '###': cv2.drawContours(ignore, points, 0, 1, -1); continue  # ignore instance
                cv2.drawContours(instance, points, 0, idx + 1, -1)

        img_paths.append(img_path)
        imgs.append(img)
        instances.append(instance)
        ignores.append(ignore)

    return [img_paths, imgs, instances, ignores]


def parse_Synth(dataset_name, split, data_dir, gt_dir):
    img_paths, boxes, texts = [], [], []

    gt_path = os.path.join(gt_dir, 'gt.mat').replace('\\', '/')
    gt_data = scio.loadmat(gt_path)

    img_files = gt_data['imnames'][0]
    gt_boxes = gt_data['wordBB'][0]
    gt_texts = gt_data['txt'][0]

    # 1. 开始遍历每张图片
    for img_file, gb, gt in zip(img_files, gt_boxes, gt_texts):
        img_file = img_file[0]
        img_path = os.path.join(data_dir, img_file).replace('\\', '/')

        img_paths.append(img_path)
        boxes.append(gb)
        texts.append(gt)

    return [img_paths, boxes, texts]


def parse_label(dataset_name, split, data_dir, gt_dir):
    if dataset_name == 'MSRA-TD500':
        return parse_MSRA(dataset_name, split, data_dir, gt_dir)
    elif dataset_name == 'TotalText':
        return parse_TT(dataset_name, split, data_dir, gt_dir)
    elif dataset_name == 'CTW1500':
        return parse_CTW(dataset_name, split, data_dir, gt_dir)
    elif dataset_name == 'ICDAR2015':
        return parse_IC15(dataset_name, split, data_dir, gt_dir)
    elif dataset_name == 'SynthText':
        return parse_Synth(dataset_name, split, data_dir, gt_dir)
