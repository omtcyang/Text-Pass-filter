import os
import os.path as osp
from dotted.collection import DottedDict


class R18_BASE:
    # 项目root路径
    proj_root = osp.dirname(osp.dirname(osp.abspath(__file__)))

    # 数据集 root路径
    data_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))

    # 设置 日志、推理结果、以及 训练权重 存放路径  
    # 由子类实现
    def log_result_checkpoint_dirs(self, cfg_name):
        log_dir = os.path.join(self.proj_root, 'outputs/log_' + cfg_name + '/')  # 日志存放路径
        if not osp.exists(log_dir):
            os.mkdir(log_dir)

        result_dir = os.path.join(self.proj_root, 'outputs/submit_' + cfg_name + '/')  # 推理结果存放路径
        if not osp.exists(result_dir):
            os.mkdir(result_dir)

        checkpoint_dir = osp.join(self.proj_root, 'checkpoints/' + cfg_name + '/')  # 训练权重存放路径
        if not osp.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        return log_dir, result_dir, checkpoint_dir

    # 设置 数据加载路径
    # 由子类实现
    def data_paths(self, dataset_name):
        train_path = DottedDict({'img': self.data_root + '/data/' + dataset_name + '/train/Image/',  # 训练数据存放路径
                                 'gt': self.data_root + '/data/' + dataset_name + '/train/GT/', })
        test_path = DottedDict({'img': self.data_root + '/data/' + dataset_name + '/test/Image/',  # 测试数据存放路径
                                'gt': self.data_root + '/data/' + dataset_name + '/test/GT/', })
        return train_path, test_path

    # 模型配置参数
    model = DottedDict(dict(
        type='First2023',  # 模型类名
        count='1',
        backbone=dict(
            type='resnet18_3',  # 模型主干网络类别
            pretrained=False,  # 模型主干网络是否需要加载与训练权重
        ),
        neck=dict(
            type='FPNSolo',  # 模型数据融合部分类型
            in_channels=(64, 128, 256, 512),  # 模型数据融合部分各个输入的通道数， resnet18
            reduce_channels=128,  # 模型数据融合部分各个输入通道的下降数
            out_channels=128,  # 模型数据融合部分各个输入通道的最终输出数
        ),
        head=dict(
            type='Decoder',  # 模型检测头的类型
            test_cfg=None,
            in_channels=128 * 4,  # 模型检测头的输入通道数，这个在代码里与neck的输出动态绑定
            hidden_dim=64,  # 模型检测头的中间隐藏通道数，resnet18
            loss_center=dict(  # 模型检测头对应的损失函数
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0
            ),
            loss_instance=dict(  # 模型检测头对应的损失函数
                type='DiceLoss',
                use_sigmoid=True,
                loss_weight=1.0
            ),
        ),
        featurehead=dict(
            type='FeatureHead',  # 模型检测头的类型
            in_channels=128 * 4,  # 模型检测头的输入通道数，这个在代码里与neck的输出动态绑定
            hidden_dim=64
        ),
    ))
