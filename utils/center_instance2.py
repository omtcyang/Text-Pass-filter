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
import argparse
import sys
sys.path.append("/mnt/cyang_text/solotext")
import time
import json
from mmcv import Config

from dataset import build_dataset
from models import build_model
from models.utils import fuse_module
import pandas as pd
from utils import ResultFormat, AverageMeter

from ctw_eval_all import compute_prh as ctw_prh
from msra_eval_all import compute_prh as msra_prh


def test(test_loader, model, cfg):
    print('Testing with pretrained model')
    contours_count = []
    contains = []
    IOU = []

    img_names = []
    gt_counts = []
    center_counts = []
    gt_covers = []
    center_covers = []

    for idx, data in enumerate(test_loader):
        sys.stdout.flush()
        data['imgs'] = data['imgs'].cuda()
        instance = data['gt_instance'][0]
        with torch.no_grad():
            H, W, count, contain, seg_masks = model.getHW2(**data)

        contours_count.extend(count)
        contains.extend(contain)

        gt_count = len(torch.unique(instance)) - 1
        center_count = len(H)

        if len(H) != 0:
            a = np.zeros(len(torch.unique(instance)) - 1)
            b = np.zeros(len(H))
            instance_index = np.zeros(len(H))
            for u_index, u in enumerate(torch.unique(instance)[1:]):
                ins = (instance == u).int()
                for index, pos in enumerate(zip(H, W)):
                    y, x = pos
                    if b[index] == 0 and ins[int(y), int(x)] == 1:
                        a[u_index] = 1
                        b[index] = 1
                        instance_index[index] = u

            gt_cover = np.sum(a)
            center_cover = np.sum(b)

            iou = []
            for i, mask in enumerate(seg_masks.cpu()):
                if b[i] == 0:
                    iou.append(0)
                else:
                    ins = (instance == instance_index[i]).int()
                    inter = torch.sum(ins * mask)
                    union = torch.sum(ins) + torch.sum(mask) - inter
                    iou.append(float(inter / union))
            IOU.extend(iou)
        else:
            gt_cover = 0
            center_cover = 0

        img_names.append(data['imgs_meta']['img_path'][0].split('/')[-1])
        gt_counts.append(gt_count)
        center_counts.append(center_count)
        gt_covers.append(gt_cover)
        center_covers.append(center_cover)

    data = np.concatenate([np.array(contours_count)[:, np.newaxis], np.array(contains)[:, np.newaxis], np.array(IOU)[:, np.newaxis]], axis=1)
    df = pd.DataFrame(data=data)
    df.to_csv('./mask.csv')

    data = np.concatenate([np.array(img_names)[:, np.newaxis],
                           np.array(gt_counts)[:, np.newaxis],
                           np.array(gt_covers)[:, np.newaxis],
                           np.array(center_counts)[:, np.newaxis],
                           np.array(center_covers)[:, np.newaxis]], axis=1)
    df = pd.DataFrame(data=data)
    df.to_csv('./center.csv')


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
    cfg.test_cfg.score_thr = 0.11
    cfg.test_cfg.mask_thr = 0.3
    cfg.test_cfg.min_score = 0.85

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

    epoch = 564
    print('▊' * 30 + ' Current Testing Epoch :' + str(epoch) + ' ' + '▊' * 30)

    checkpoint_path = osp.join(cfg.checkpoint_dir + 'twelfth', 'checkpoint_' + str(epoch) + 'ep.pth.tar')
    main(checkpoint_path, test_loader, cfg)
