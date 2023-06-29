import numpy as np
import cv2
from PIL import Image
import os
import os.path as osp


#### save_img_results
def save_img_results(datas, path):
    print('save_img_results')

    tails = ['img', 'ignore', 'instance', 'center', 'centernesses', 'dminimum']
    for idx, img in enumerate(datas):
        if idx == 0:
            img_Image = Image.fromarray(img)
        else:
            img_Image = Image.fromarray(img * 255)
        if img_Image.mode == 'F':
            img_Image = img_Image.convert('RGB')
        root = '/mnt/cyang_text/first2023/dataset/datasets/visualization/'
        filepath = (root + tails[idx] + '_' + path.split('/')[-1]).replace('.jpg', '.png')
        img_Image.save(filepath)


#### save_text_results
def save_text_results(datas, path):
    print('save_text_results')

    root = '/mnt/cyang_text/first2023/dataset/datasets/txtlog/'
    path = root + 'label_information.txt'
    tails = ['img', 'ignore', 'instance', 'center', 'centernesses', 'dminimum']
    with open(path, mode='a+') as writer:
        for i in range(1, len(datas)):
            writer.write('np.unique' + '(' + tails[i] + ')==>' + str(np.unique(datas[i])) + '\n')


#### check the position of error
def check():
    try:
        codes
    except Exception as e:
        print(e)
        print(sys.exc_info())
        print('\n', '>>>' * 20)
        print(traceback.print_exc())
        print('\n', '>>>' * 20)
        print(traceback.format_exc())
