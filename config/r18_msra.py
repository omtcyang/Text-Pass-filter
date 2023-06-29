from dotted.collection import DottedDict

from .r18_base import R18_BASE


class R18_MSRA(R18_BASE):
    # 训练条件参数设置           
    cfg_name = 'r18_msra'  # 配置文件名称
    dataset_name = 'MSRA-TD500'  # 数据集简称
    report_speed = False  # 是否打印速度

    # 日志、推理结果、以及 训练权重 存放路径
    log_dir, result_dir, checkpoint_dir = R18_BASE.log_result_checkpoint_dirs(R18_BASE, cfg_name)

    # 数据路径
    train_path, test_path = R18_BASE.data_paths(R18_BASE, dataset_name)

    # 标签生成参数
    data = DottedDict(dict(
        train=dict(
            type='OnlineDataset',  # 训练数据生成过程中，训练数据加载类的名称
            dataset_name=dataset_name,
            data_path=train_path,
            split='train',  # 训练数据生成过程中，是否是训练集
            short_size=736,  # 训练数据生成过程中，将短边缩放至 short_size
            sample_strides={'4': 1, '8': 0, '16': 0, '32': 0},  # 训练数据生成过程中，center、centerness 以及 dminimum 的label尺寸
            sample_types={'single': 0, 'multi': 1},  # 训练数据生成过程中，对中心点的采样方式，是 1.只对中心点采样，还是 2.周围多个点
            mask_size=4,  # 训练数据生成过程中，instance和ignore的尺寸
            is_ignore=True,  # 训练数据生成过程中，该数据集是否包含 ignore 样本
            is_transform=True,  # 训练数据生成过程中，是否需要数据增强
        ),
        test=dict(
            type='OnlineDataset',  # 测试数据生成过程中，测试数据加载类的名称
            dataset_name=dataset_name,
            data_path=test_path,
            split='test',  # 测试数据生成过程中，是否是训练集
            short_size=736,  # 测试数据生成过程中，将短边缩放至 short_size
            is_ignore=True,  # 测试数据生成过程中，该数据集是否包含 ignore 样本
            is_transform=False,  # 测试数据生成过程中，是否需要数据增强
            report_speed=report_speed,  # 测试数据生成过程中，是否需要测试速度
        )
    ))

    # 训练过程，梯度优化配置
    train_cfg = DottedDict(dict(
        visualize=False,  # 训练数据生成过程中，是否需要可视化
        batch_size=16,
        lr=1e-3,  # 训练过程，学习率的初始值
        schedule='polylr',  # 训练过程，学习率变化方式
        start_epoch=0,  # 训练过程，初始学习epoch
        epoch=1200,  # 训练过程，目标学习epoch
        optimizer='Adam',  # 训练过程，优化器的类型
        is_resume=True,  # 是否需要resume
        resume='./checkpoints/' + cfg_name + '/checkpoint.pth.tar'  # resume权重路径
    ))

    # 推理过程，超参配置 12th
    test_cfg = DottedDict(dict(
        nms_pre=500,
        score_thr=0.13,
        mask_thr=0.45,
        matrix_thr=0.95,
        # update_thr=0.04,
        kernel='gaussian',  # gaussian/linear
        sigma=2.0,
        max_per_img=50,
        min_score=0.9,  # 推理过程，热力图最小平均置信度
        min_area=10,  # 推理过程，热力图最小面积
        report_speed=True,  # 推理过程，是否需要打印速度
        bbox_type='rect',  # 推理过程，目标文字实例形状
    ))
    R18_BASE.model.head.test_cfg = test_cfg  # 实现父类 model ==> head ==> 的test_cfg属性

    # 打印配置文件信息
    def display(self):
        config_info = ''
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                config_info += '===>' + "{:20} {}".format(a, getattr(self, a))
                config_info += '\n'
                # print('===>',"{:20} {}".format(a, getattr(self, a)))
        return config_info

# r = R18_MSRA()
# print(dir(r))
# print(r.train_path.img)
