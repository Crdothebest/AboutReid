class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 计数器，记录更新次数

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val  # 更新当前值
        self.sum += val * n  # 更新总和，乘以数量n
        self.count += n  # 更新计数器
        self.avg = self.sum / self.count  # 更新平均值
