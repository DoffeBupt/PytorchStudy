## 第七章 神经网络与全连接层 
### Chap7-1 逻辑回归
- 不咋用了，逻辑指的是输出为0，1，回归指的是最开始的时候用的是MSE，思想类似于回归
- 不用ACC优化是因为微小的prob预测，不足以顶到阈值以外，让0—>1，因而acc未变，grad为0，容易出问题
- softmax
    - 把所有输出归一化
    - 让大的比小的的优势更大

### Chap7-2 交叉熵
- 衡量两个分布的差距的函数理论上应该为K-L散度
- 当其中一个分布为one-hot的时候，交叉熵等价于K-L散度
- 交叉熵多用于多分类问题，与softmax结合
- softmax不爱用MSE主要是因为，softmax在输入很大的时候，都会输出1附近的值，梯度不好传回去
- 如果错误的情况下，交叉熵在另外一侧会有一个很巨大的loss，方便传回去
- torch.nn.functional的cross_entropy中，传入的为softmax之前的x矩阵
- 第二个值为one-hot中编码为1的索引
- 实现的时候，源码中实际上并没有咋算交叉熵，主要都在算softmax+log
- cross_entropy = softmax + log + nil_loss

### Chap7-3 多分类问题
- 初始化十分的重要，直接影响到结果
    - 可以试试torch.nn.init.kaiming_normal_(w1)
- 最后一层输出以前，可以考虑不加激活函数，因为算交叉熵之前一定会过一下softmax，然后交叉熵

### Chap7-4 全连接层
- nn.Linear
    - 直接内部带着W和b以及他们的梯度啥的了，不用管了
    - 结合nn.Sequential一起使用
    - nn.Sequential里边传入一堆linear和relu就行
    - relu记得(inplace=True)，这样相当于传入的东西直接被赋值了
    - 整个网络直接继承nn.Module就行
- 看了一些源码，有些小笔记
    - nn.Sequential()是把每一个扔进去的元素进行call()的那种运算
    - nn.Module()的实例直接用，和直接xx.forward()是一致的
    
### Chap7-5 一些激活函数和GPU加速
- 一些激活函数
    - sigmoid, 怼到[0,1]，x绝对值较大的时候容易梯度消失
    - tanh, 瘦高版本的sigmoid，在RNN中常见
    - RELU, 负0，正x，不容易梯度消失
    - Leaky ReLu, 负的为一个-ax(a 一般为0.02)
    - SELU relu+指数函数 圆润版本
    - softplus 圆润版RELU
- GPU加速
    - .cuda为老版本，容易出各种问题
    - 新版本定义device('cuda:1,2')
    - x.to(device) 方便一次性改一大堆
    - 一般是net，输入，输出，算loss的函数要放到GPU
    
### Chap7-6 测试
- 主要是为了防止过拟合
- 一般一个epoch啥的搞一下
- 主要用的思路就是argmax()+torch.eq(预测,label)/len(预测)

### Chap 7-7 可视化
- 有TensorboardX和Visdom
- 主要的逻辑都是把数据扔到一个网页中
- TensorboardX的缺点主要是需要不停地写入文件，而且需要从tensor转到numpy
- visdom貌似是亲儿子，不需要转换，而且不用写入文件
- visdom占用8097口，TensorboardX6006
- 常用命令
    - 初始化viz = Visdom()
    - 一条线viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    - 两条线viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.',
                                                     legend=['loss', 'acc.']))
    - 更新线，append加在末尾viz.line([[test_loss, correct / len(test_loader.dataset)]],
                   [global_step], win='test', update='append')
    - 画图viz.images(data.view(-1, 1, 28, 28), win='x')
    - 画文本viz.text(str(pred.detach().cpu().numpy()), win='pred',
                   opts=dict(title='pred'))  
- 注意，win有点像ID的感觉，一般来说，每一次val反向传播前算的loss就可以顺手扔到visdom里更新一下
- 每一次用验证集eval的时候，可以更新那些乱七八糟的梯度以及验证loss