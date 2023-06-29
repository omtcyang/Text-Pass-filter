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
from utils import ResultFormat, AverageMeter

from ctw_eval_all import compute_prh as ctw_prh
from msra_eval_all import compute_prh as msra_prh


def test(test_loader, model, cfg):
    res = {'less': [], 'more': [], 'both': []}
    print('Testing with pretrained model')
    contours_count = []
    contains = []
    for idx, data in enumerate(test_loader):
        sys.stdout.flush()
        # prepare input
        data['imgs'] = data['imgs'].cuda()
        instance = data['gt_instance'][0]
        with torch.no_grad():
            HW, con, conta = model.getHW(**data)
        contours_count.extend(con)
        contains.extend(conta)
        if HW is None:
            if len(torch.unique(instance)) == 1:
                continue
            res['less'].append(data['imgs_meta']['img_path'][0].split('/')[-1])
        else:
            if len(torch.unique(instance)) == 1:
                res['more'].append(data['imgs_meta']['img_path'][0].split('/')[-1])
            else:
                a = np.zeros(len(torch.unique(instance)) - 1)
                b = np.zeros(len(HW))
                for u_index, u in enumerate(torch.unique(instance)[1:]):
                    ins = (instance == u).int()
                    for index, pos in enumerate(HW):
                        y, x = pos
                        if b[index] == 0 and ins[int(y), int(x)] == 1:
                            b[index] = 1
                            a[u_index] = 1

                sum_a = np.sum(a)
                sum_b = np.sum(b)
                if sum_a == len(a) and sum_b != len(b):
                    # 预测中心点多了，实例之外还存在点
                    res['more'].append(data['imgs_meta']['img_path'][0].split('/')[-1])
                elif sum_a != len(a) and sum_b == len(b):
                    # 预测中心点少了，有的实例没有覆盖
                    res['less'].append(data['imgs_meta']['img_path'][0].split('/')[-1])
                elif sum_a != len(a) and sum_b != len(b):
                    # 两种情况都存在
                    res['both'].append(data['imgs_meta']['img_path'][0].split('/')[-1])

    import pandas as pd

    data = np.concatenate([np.array(contours_count)[:, np.newaxis], np.array(contains)[:, np.newaxis]], axis=1)
    df = pd.DataFrame(data=data)
    df.to_csv('./contours_contains.csv')

    df2 = pd.DataFrame(res['less'])
    df3 = pd.DataFrame(res['more'])
    df4 = pd.DataFrame(res['both'])
    writer = pd.ExcelWriter('center.xlsx')
    df2.to_excel(excel_writer=writer, sheet_name='less')
    df3.to_excel(excel_writer=writer, sheet_name='more')
    df4.to_excel(excel_writer=writer, sheet_name='both')
    writer.save()


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

    epoch = 564
    print('▊' * 30 + ' Current Testing Epoch :' + str(epoch) + ' ' + '▊' * 30)

    checkpoint_path = osp.join(cfg.checkpoint_dir+'twelfth', 'checkpoint_' + str(epoch) + 'ep.pth.tar')
    main(checkpoint_path, test_loader, cfg)
