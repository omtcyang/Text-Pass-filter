import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import os.path as osp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import numpy as np
import random
import argparse
import sys
import time
import json
from mmcv import Config
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from dataset import build_dataset
from models import build_model
from utils import AverageMeter

# 随机种子设置
torch.manual_seed(300)  # cpu
torch.cuda.manual_seed(300)  # gpu
torch.cuda.manual_seed_all(300)  # all gpu
torch.backends.cudnn.deterministic = True  # forward backward
np.random.seed(300)  # np
random.seed(300)  # random


def train(train_loader, model, optimizer, writer, epoch, cfg):
    model.module.train()

    # meters 用于存放训练过程中loss的中间值
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_instance = AverageMeter()
    losses_center = AverageMeter()
    losses_classification = AverageMeter()
    losses_fuse_gt = AverageMeter()

    # start time
    start = time.time()
    for iter, data in enumerate(train_loader):
        # adjust learning rate
        adjust_learning_rate(optimizer, train_loader, epoch, iter, cfg)

        # forward
        outputs = model(**data)

        # detection loss
        loss_instance = torch.mean(outputs['loss_mask'])
        losses_instance.update(loss_instance.item())

        loss_center = torch.mean(outputs['loss_cates'])
        losses_center.update(loss_center.item())

        loss_classification = torch.mean(outputs['loss_classification'])
        losses_classification.update(loss_classification.item())

        loss_fuse_gt = torch.mean(outputs['loss_fuse_gt'])
        losses_fuse_gt.update(loss_fuse_gt.item())

        loss = loss_instance + loss_center + loss_classification +loss_fuse_gt*0.7

        losses.update(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)

        # update start time
        start = time.time()

        # print log
        if iter % 5 == 0:
            print('===> Cur_Loss: instance:%.3f, center:%.3f, classification:%.3f, fuse_gt:%.3f, loss:%.3f'
                  % (losses_instance.val[-1], losses_center.val[-1], losses_classification.val[-1], losses_fuse_gt.val[-1], losses.val[-1]))
            global_step = epoch * len(train_loader) + iter
            writer.add_scalar('train/loss_instance', losses_instance.avg, global_step)
            writer.add_scalar('train/loss_center', losses_center.avg, global_step)
            writer.add_scalar('train/loss_classification', losses_classification.avg, global_step)
            writer.add_scalar('train/loss_fuse_gt', losses_fuse_gt.avg, global_step)
            writer.add_scalar('train/loss', losses.avg, global_step)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)


def adjust_learning_rate(optimizer, dataloader, epoch, iter, cfg):
    schedule = cfg.train_cfg.schedule

    if isinstance(schedule, str):
        assert schedule == 'polylr', 'Error: schedule should be polylr!'
        cur_iter = epoch * len(dataloader) + iter
        max_iter_num = cfg.train_cfg.epoch * len(dataloader)
        lr = cfg.train_cfg.lr * (1 - float(cur_iter) / max_iter_num) ** 0.9

    elif isinstance(schedule, tuple):
        lr = cfg.train_cfg.lr
        for i in range(len(schedule)):
            if epoch < schedule[i]:
                break
            lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, checkpoint_dir, cfg):
    file_path = osp.join(checkpoint_dir, 'checkpoint.pth.tar')
    torch.save(state, file_path)

    if (cfg.data.train.type in ['synth']) or (state['iter'] == 0):
        file_name = 'checkpoint_%dep.pth.tar' % state['epoch']
        file_path = osp.join(checkpoint_dir, file_name)
        torch.save(state, file_path)


def main(cfg):
    ### step 1. 构建权重存放路径 以及 日志记录器
    checkpoint_dir = osp.join('checkpoints', cfg.cfg_name)  # 权重
    writer = SummaryWriter(log_dir=cfg.log_dir)  # 日志

    ### step 2. 构建 model
    logger.info('Build model')
    model = build_model(cfg.model)
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

    ### step 3. 构建数据加载器
    logger.info('Build data loader')
    dataset = build_dataset(cfg.data.train)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train_cfg.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=False
    )

    # ### step 4. 构建优化器
    logger.info('Build optimizer')
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        if cfg.train_cfg.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train_cfg.lr, momentum=0.99, weight_decay=5e-4)
        elif cfg.train_cfg.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_cfg.lr)

    # ### step 5. 是否进行resume训练
    if cfg.train_cfg.is_resume:
        assert osp.isfile(cfg.train_cfg.resume), 'Error: no checkpoint directory found!'
        logger.info('Resuming... %s.' % cfg.train_cfg.resume)
        checkpoint = torch.load(cfg.train_cfg.resume)
        cfg.train_cfg.start_epoch = checkpoint['epoch']

        d = dict()
        for key, value in checkpoint['state_dict'].items():
            tmp = 'module.' + key
            d[tmp] = value
        model.load_state_dict(d)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        logger.info('Not Resume')

    # ### step 6. 开始训练
    logger.info('Start to training')
    for epoch in range(cfg.train_cfg.start_epoch, cfg.train_cfg.epoch):
        logger.info('Epoch: [%d | %d]' % (epoch + 1, cfg.train_cfg.epoch))

        train(train_loader, model, optimizer, writer, epoch, cfg)

        if epoch + 1 >= 30:
            state = dict(
                epoch=epoch + 1,
                iter=0,
                state_dict=model.module.state_dict(),
                optimizer=optimizer.state_dict()
            )
            save_checkpoint(state, checkpoint_dir, cfg)


if __name__ == '__main__':
    # DEBUG < INFO < WARNING < ERROR < CRITICAL
    from config import r18_msra
    import warnings

    warnings.filterwarnings('ignore')

    config_info = r18_msra.display(r18_msra)

    logger.info('Load config')
    logger.info(config_info)

    main(r18_msra)
