import os
import os.path as osp

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import numpy as np
import argparse
import sys


import time
import json
from mmcv import Config

from dataset import build_dataset
from models import build_model
from models.utils import fuse_module
from utils import ResultFormat, AverageMeter

from ctw_eval_all import compute_prh as ctw_prh
from msra_eval_all import compute_prh as msra_prh


def report_speed(outputs, speed_meters):
    total_time = 0
    for key in outputs:
        if 'time' in key:
            total_time += outputs[key]
            speed_meters[key].update(outputs[key])
            print('%s: %.4f' % (key, speed_meters[key].avg))

    speed_meters['total_time'].update(total_time)
    print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))


def test(test_loader, model, cfg):
    rf = ResultFormat(cfg.data_type, cfg.test_cfg.result_path)

    if cfg.data.test.report_speed:
        speed_meters = dict(
            backbone_time=AverageMeter(500),
            neck_time=AverageMeter(500),
            det_head_time=AverageMeter(500),
            det_post_time=AverageMeter(500),
            total_time=AverageMeter(500)
        )

    print('Testing with pretrained model')
    for idx, data in enumerate(test_loader):
        # print('Testing %d/%d' % (idx, len(test_loader)))
        sys.stdout.flush()

        # prepare input
        data['imgs'] = data['imgs'].cuda()

        # forward
        with torch.no_grad():
            outputs = model(**data)

        if cfg.data.test.report_speed:
            report_speed(outputs, speed_meters)

        # save result
        image_name, _ = osp.splitext(osp.basename(test_loader.dataset.img_paths[idx]))
        rf.write_result(image_name, outputs)


def main(checkpoint_path, test_loader, cfg):
    sys.stdout.flush()

    # model
    model = build_model(cfg.model)
    # model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    # model.module.eval()
    model = model.cuda()
    model.eval()

    if checkpoint_path is not None:
        if osp.isfile(checkpoint_path):
            print("Loading model and optimizer from checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            raise

    # fuse conv and bn
    model = fuse_module(model)

    # test
    test(test_loader, model, cfg)


if __name__ == '__main__':

    from config import r18_msra
    import warnings

    warnings.filterwarnings('ignore')

    cfg_dict = {'msra': r18_msra}
    # , 'tt': r18_tt, 'ic15': r18_ic15, 'r18_ctw': r18_ctw}
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
        num_workers=1,
        pin_memory=False
    )

    # 开始测试
    p, r, h, e = np.array([]), np.array([]), np.array([]), np.array([])
    for epoch in np.arange(440, 1200, 1):

        print('▊' * 30 + ' Current Testing Epoch :' + str(epoch) + ' ' + '▊' * 30)

        checkpoint_path = osp.join(cfg.checkpoint_dir, 'checkpoint_' + str(epoch) + 'ep.pth.tar')
        checkpoint_nxt_path = osp.join(cfg.checkpoint_dir, 'checkpoint_' + str(epoch + 1) + 'ep.pth.tar')
        while 1:
            if not osp.exists(checkpoint_nxt_path):
                time.sleep(300)
            else:
                break

        main(checkpoint_path, test_loader, cfg)

        print('Computing performance')
        # 1. MSRA-TD500
        if cfg.data_type == 'MSRA':
            pred_root = cfg.test_cfg.result_path
            gt_root = cfg.test_path.gt
            precision, recall, hmean = msra_prh(pred_root, gt_root)

        # 2. Total-Text
        elif cfg.data_type == 'TT':
            cmd = 'cd eval && cd tt && python Deteval.py'
            res_cmd = os.popen(cmd)
            res_cmd = res_cmd.read()
            precision = float(res_cmd.split('_')[1])
            recall = float(res_cmd.split('_')[9].split('/')[0])
            hmean = float(res_cmd.split('_')[-1])

        # 3. CTW1500
        elif cfg.data_type == 'CTW':
            pred_root = cfg.result_path
            gt_root = cfg.test_path.gt
            precision, recall, hmean = ctw_prh(pred_root, gt_root)

        # 4. ICDAR2015
        elif cfg.data_type == 'IC15':
            cmd = 'cd eval && cd ic15 && python script.py -g=gt.zip -s=../../outputs/submit_ic15.zip'
            res_cmd = os.popen(cmd)
            res_cmd = res_cmd.read()
            precision = float(res_cmd.split(',')[0].split(':')[-1])
            recall = float(res_cmd.split(',')[1].split(':')[-1])
            hmean = float(res_cmd.split(',')[2].split(':')[-1])

        if len(h) < 5:
            p = np.insert(p, 0, precision)
            r = np.insert(r, 0, recall)
            h = np.insert(h, 0, hmean)
            e = np.insert(e, 0, epoch)

        else:
            minIndex = np.argmin(h)
            if hmean > h[minIndex]:
                # 删除权重
                # checkpoint_del = osp.join(cfg.checkpoint_dir, 'checkpoint_' + str(int(e[minIndex])) + 'ep.pth.tar')
                # os.remove(checkpoint_del)

                p = np.delete(p, minIndex)
                r = np.delete(r, minIndex)
                h = np.delete(h, minIndex)
                e = np.delete(e, minIndex)

                p = np.insert(p, 0, precision)
                r = np.insert(r, 0, recall)
                h = np.insert(h, 0, hmean)
                e = np.insert(e, 0, epoch)
            # else:
            #     checkpoint_del = osp.join(cfg.checkpoint_dir, 'checkpoint_' + str(int(epoch)) + 'ep.pth.tar')
            #     os.remove(checkpoint_del)
            # pass

        for ee, pp, rr, hh in zip(e, p, r, h):
            print(str(ee) + ' : ' + 'p: %.6f, r: %.6f, f: %.6f' % (pp, rr, hh))
