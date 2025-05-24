class TrainCfg:
    """配置类，通过类属性管理参数"""
    # 数据集
    DataRoot = "../data"  # 数据集根目录
    BatchSize = 128  # 批量大小
    num_workers = 4  # 数据加载线程数
    seed = 42
    ratio = 0.5
    # 超参数
    Epoch = 120
    LearningRate = 0.01
    MileStones = [40, 80]

    # log
    SaveCheckpoint = 10