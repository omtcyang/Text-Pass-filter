#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time : 2023/1/6 15:12
@Author : Haozhao Ma
@Email : haozhaoma@mail.nwpu.edu.cn
@time: 2023/1/6 15:12
"""
import os
import os.path as osp

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import numpy as np
import sys
sys.path.append("/mnt/cyang_text/solotext")

from dataset import build_dataset
from models import build_model
from models.utils import fuse_module
import pandas as pd
import cv2
import cc_torch


def test(test_loader, model, cfg):
    print('Testing with pretrained model')

    imgs = []
    index1 = []
    index2 = []
    IOU = []

    for idx, data in enumerate(test_loader):
        os.mkdir(os.path.join('img', data['imgs_meta']['img_path'][0].split('/')[-1]))
        os.mkdir(os.path.join('img', data['imgs_meta']['img_path'][0].split('/')[-1], '1'))
        os.mkdir(os.path.join('img', data['imgs_meta']['img_path'][0].split('/')[-1], '0'))

        sys.stdout.flush()
        data['imgs'] = data['imgs'].cuda()
        instance = data['gt_instance'][0]
        gt_count = len(torch.unique(instance)) - 1
        with torch.no_grad():
            H, W, seg_masks = model.getAllMask(**data)
        if len(H) != 0:
            b = np.zeros(len(H))
            gt_index = np.full(len(H), -1)
            mask_index = np.full(len(H), -1)
            for u_index, u in enumerate(torch.unique(instance)[1:]):
                ins = (instance == u).int()
                for index, pos in enumerate(zip(H, W)):
                    y, x = pos
                    if b[index] == 0 and ins[int(y), int(x)] == 1:
                        b[index] = 1
                        gt_index[index] = u_index

            index1.extend(gt_index)

            for i in range(len(H)):
                img = cv2.cvtColor(data['imgs_ori'][0].numpy(), cv2.COLOR_RGB2BGR)
                img[np.nonzero(seg_masks[i].cpu().numpy())] = (0, 255, 0)
                cv2.circle(img, (int(W[i]), int(H[i])), 5, (0, 0, 255), thickness=10)
                if b[i] == 1:
                    cv2.imwrite(os.path.join('img', data['imgs_meta']['img_path'][0].split('/')[-1], '1/{}.jpg'.format(i)), img)
                else:
                    cv2.imwrite(os.path.join('img', data['imgs_meta']['img_path'][0].split('/')[-1], '0/{}.jpg'.format(i)), img)

                imgs.append(data['imgs_meta']['img_path'][0].split('/')[-1])
                connect_label = cc_torch.connected_components_labeling(seg_masks[i])
                mask_count = len(torch.unique(connect_label)) - 1
                iou = np.zeros((mask_count, gt_count))

                for j, m_u in enumerate(torch.unique(connect_label)[1:]):
                    mask = (connect_label == m_u).cpu().int()
                    if connect_label[int(H[i]), int(W[i])] == m_u:
                        mask_index[i] = j
                    for k, g_u in enumerate(torch.unique(instance)[1:]):
                        ins = (instance == g_u).int()
                        inter = torch.sum(ins * mask)
                        union = torch.sum(ins) + torch.sum(mask) - inter
                        iou[j, k] = float(inter / union)

                IOU.append(iou)

            index2.extend(mask_index)
    for i in range(len(imgs)):
        row = IOU[i].shape[0]
        col = IOU[i].shape[1]
        a = {}
        imgs_list = [imgs[i]]
        index1_list = [index1[i]]
        index2_list = [index2[i]]
        for j in range(row - 1):
            imgs_list.append(np.nan)
            index1_list.append(np.nan)
            index2_list.append(np.nan)
        a['imgs'] = imgs_list
        a['index1'] = index1_list
        a['index2'] = index2_list
        for j in range(col):
            a['iou{}'.format(j)] = IOU[i][:, j]
        df = pd.DataFrame(a)
        df.to_csv('1.csv', mode='a', header=True, index=False)


def main(checkpoint_path, test_loader, cfg):
    sys.stdout.flush()
    model = build_model(cfg.model)
    model = model.cuda()
    model.eval()
    if osp.isfile(checkpoint_path):
        print("Loading model and optimizer from checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
    model = fuse_module(model)
    test(test_loader, model, cfg)


if __name__ == '__main__':
    from config import r18_msra
    import warnings

    warnings.filterwarnings('ignore')

    cfg_dict = {'msra': r18_msra}
    cfg = cfg_dict['msra']
    cfg.data_type = 'MSRA'
    cfg.test_cfg.result_path = cfg.result_dir

    cfg.model.backbone.pretrained = False  # 不加载resnet的预训练权重，避免浪费时间
    cfg.report_speed = False  # 是否打印测速度

    # data loader
    data_loader = build_dataset(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    epoch = 515
    print('▊' * 30 + ' Current Testing Epoch :' + str(epoch) + ' ' + '▊' * 30)

    checkpoint_path = osp.join(cfg.checkpoint_dir + 'eleventh', 'checkpoint_' + str(epoch) + 'ep.pth.tar')
    main(checkpoint_path, test_loader, cfg)
