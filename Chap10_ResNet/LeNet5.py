import torch
# nn出来的都是对象,要实例化
from torch import nn
# F家里的都是方法,直接用就行
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
        for cifar10
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # 3层CNN+2层FNN
            # x:[b,3,32,32] -> [b,6,28,28]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        # out-> [b,16,5,5]
        # flatten
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

        # 用交叉熵,自动附赠一个softmax
        self.criteon = nn.CrossEntropyLoss()

    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        batchsz = x.size(0)
        # CNN
        x = self.conv_unit(x)  # nn家里的东西,直接layer()就等价于用了forward方法
        x = x.view(batchsz, -1)  # 第一个维度变为batchsize,剩下维度他自己算
        # FC
        # logits 输出 [b,10]
        logits = self.fc_unit(x)  # logits一般指的是送入softmax之前的那个玩意
        # pred = F.softmax(logits, dim=1) CE附赠softmax,吃softmax之前那个
        return logits


def main():
    net = LeNet5()
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print(out.shape)


if __name__ == '__main__':
    main()
