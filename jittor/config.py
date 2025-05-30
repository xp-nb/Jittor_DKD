class TrainCfg:
    """配置类，通过类属性管理参数"""
    DataRoot = "../data"  # 数据集根目录
    BatchSize = 128  # 批量大小
    num_workers =4 #数据加载线程数
    seed = 42

    # 超参数
    # Teacher
    # ratio = 0.5
    # Epoch = 120
    # LearningRate = 0.001
    # MileStones = [40, 80]
    # TeacherDir = "../config/teacher/teacher.pkl"
    # Temperature = 4.0
    # Student
    ratio = 0.1
    Epoch = 40
    LearningRate = 0.001
    MileStones = [10, 20]
    TeacherDir = "../config/teacher/teacher.pkl"
    Temperature = 4.0
    # log
    SaveCheckpoint = 10