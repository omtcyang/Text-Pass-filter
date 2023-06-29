#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time : 2023/1/1 16:42
@Author : Haozhao Ma
@Email : haozhaoma@foxmail.com
@time: 2023/1/1 16:42
"""
import sys
sys.path.append("/mnt/cyang_text/solotext")
from config import r18_msra
import os
from dataset.datasets.parser import parse_MSRA
import cv2

img = r18_msra.test_path.img
gt = r18_msra.test_path.gt

out_path = '/mnt/cyang_text/solotext/outputs/submit_r18_msra/gt'
label = parse_MSRA(None, 'test', img, gt)
path = label[0]
ins = label[2]
for p, i in zip(path, ins):
    name = p.split('/')[-1]
    cv2.imwrite(os.path.join(out_path, name), i * 255)
