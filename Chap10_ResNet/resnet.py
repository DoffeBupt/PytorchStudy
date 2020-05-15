import torch
from torch import nn
import torch.nn.functional as F


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        '''

        :param ch_in:
        :param ch_out:
        '''
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.AvgPool2d(kernel_size=stride, stride=stride)
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),  # 这里别忘了,短接也得搞一下hhh
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        # x->conv1->bn->relu
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # shortCut
        # element-wise add
        # 注意,ch_in应该与ch_out一致
        # extra layer主要就干的这个事情
        extraX = self.extra(x)
        out = extraX + out
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        # follow 4 blks
        self.blk1 = ResBlk(64, 128, 2)
        self.blk2 = ResBlk(128, 256, 2)
        self.blk3 = ResBlk(256, 512, 2)
        self.blk4 = ResBlk(512, 512, 2)

        self.outlayer = nn.Linear(512, 10)

    def forward(self, x):
        '''

        :param x:
        :return:
        '''

        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        # [b,512,x,x]->[b,512,1,1] 输入自适应,出来的肯定是个1,1
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print(x.shape)
        x = x.view(x.size(0), -1) # x.size0是batch的大小
        out = self.outlayer(x)
        return out


if __name__ == '__main__':
    tmp = torch.randn(2, 3, 32, 32)
    net = ResNet18()
    out = net(tmp)
    print(out.shape)
