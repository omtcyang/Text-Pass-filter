import os 
import os.path as osp
from dotted.collection import DottedDict

from .r18_base import R18_BASE


class R18_TT(R18_BASE):

    # 训练条件参数设置           
    cfg_name='r18_tt'  # 配置文件名称
    dataset_name='TotalText'  # 数据集简称  

    
    # 日志、推理结果、以及 训练权重 存放路径
    log_dir, result_dir, checkpoint_dir = R18_BASE.log_result_checkpoint_dirs(R18_BASE,cfg_name)


    # 数据路径
    train_path, test_path = R18_BASE.data_paths(R18_BASE,dataset_name)


    # 标签生成参数
    data = DottedDict(dict(
        train=dict(
            dataset_name=dataset_name,
            data_path=train_path,
            type='OnlineDataset',  # 训练数据生成过程中，训练数据加载类的名称
            split='train',  # 训练数据生成过程中，是否是训练集
            short_size=640,  # 训练数据生成过程中，将短边缩放至 short_size
            is_ignore=True,  # 训练数据生成过程中，该数据集是否包含 ignore 样本
            is_transform=True,  # 训练数据生成过程中，是否需要数据增强
        ),
        test=dict(
            dataset_name=dataset_name,
            data_path=test_path,
            type='OnlineDataset',  # 测试数据生成过程中，测试数据加载类的名称
            split='test',  # 测试数据生成过程中，是否是训练集
            short_size=640,  # 测试数据生成过程中，将短边缩放至 short_size
            is_ignore=True,  # 测试数据生成过程中，该数据集是否包含 ignore 样本
            is_transform=False,  # 训练数据生成过程中，是否需要数据增强
        )
    ))


    # 训练过程，梯度优化配置
    train_cfg = DottedDict(dict(
        visualize=False,  # 训练数据生成过程中，是否需要可视化
        batch_size=16,
        lr=1e-3,  # 训练过程，学习率的初始值
        schedule='polylr',  # 训练过程，学习率变化方式
        start_epoch=0,  # 训练过程，初始学习epoch
        epoch=600,  # 训练过程，目标学习epoch
        optimizer='Adam',  # 训练过程，优化器的类型
        is_resume=False,  # 是否需要resume
        resume='./checkpoints/'+cfg_name+'/checkpoint.pth.tar' # resume权重路径
    ))


    # 推理过程，超参配置
    test_cfg = DottedDict(dict(
        min_score=0.83,  # 推理过程，热力图最小平均置信度
        min_area=32,  # 推理过程，热力图最小面积
        report_speed=False,  # 推理过程，是否需要打印速度
        bbox_type='poly',  # 推理过程，目标文字实例形状
    ))
    R18_BASE.model.head.test_cfg=test_cfg  # 实现父类 model-->head-->的test_cfg属性

    # 打印配置文件信息
    def display(self):
        print("\n###########Configurations:###########")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print('===>',"{:20} {}".format(a, getattr(self, a)))
        print("\n###########Configurations:###########")


# r = R18_MSRA()
# print(dir(r))
# print(r.train_path.img)



