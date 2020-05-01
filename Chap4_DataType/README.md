## 第四章 数据类型
### 基本数据类型
- size = shape 看形状
- dim = len(size) 维度个数
- numel 有多少个元素
    - shape = 空 标量 loss之类的
    - shape = 1 向量 线性输入，bias等
    - shape = 2 batch of 线性输入
    - shape = 3 RNN input
    - shape = 4 CNN input 
- torch.rand(1,2,3) 均匀分布一个形状[1,2,3]的
- torch.randn(1,2,3) 正态分布一个...的
- torch.FloatTensor(1,2,3) 随机分布一个[1,2,3]的Float Tensor

### 创建Tensor
- 从numpy来: torch.from_numpy(a)
- 从list来: torch.tensor(list)
- 默认基本都是float
- rand 均匀分布， randn 正态分布
- *_like(a) 弄一个和a形状一样的
- randint(1,10,[2,3]) [1,10]，形状为(2,3)
- full ([2,3],7) 形状为[2,3]全是7
- arange(0,10,2) [0,10)，间隔为2,返回torch的vector
- linespace/logspace (0,10,steps=4) 均匀的[0,10]的四个数字
- logspace (0,-1,steps=4) 均匀的[0,0.1]的四个数字,base设置底数
- onse,zeros,eye 全零，全一，对角阵
- randperm(10) [0,10)的打乱的数组，常用于shuffle

### Tensor的索引与切片
- 设a.shape = [4,3,28,28]
- a[0,0,2,3] 取一个标量出来
- a[2] 取第一个维度的第二个
- 一个冒号,全部包含，左，右，俩冒号，最右边代表间隔
    - a[:2,3] 第一个维度的[0,2)，第二个维度的3
    - a[2,2,::3,1:23:2] 第四个维度的[1,23) 间隔为2
    - a[2,...,3] 第一个维度2，最后维度3，中间都取
- a.index_select(3,torch.tensor(2,3))
    - 第3个维度的第2,3号
    - 必须用Tensor
    - 等价于[:,:,[2,3],:]
- mask大法
    - mask = a.ge(0.5) 返回一个全是TF的东西，大于等于0.5
    - torch.masked_select(a,mask)
    - 返回一个vector,拍平了的
- take大法(不常用)
    - torch.take(src,torch.Tensor([0,3,5]))
    - 拍平了src,把序号035的返回
 
### Tensor维度变换
- view = reshape
    - a.view(1,3,5,6)
    - 只要元素数目一致，就可以搞，但是最好时刻跟踪住存储顺序
- Squeeze and unsqueeze
    - unsqueeze 新的增加的维度的位置在哪里
    - squeeze 把哪个的维度数目为1的维度删除，不写就是全删
    - 常和Expand等函数一起用，用于拓展bias
- Expand Repeat 假设a[1,1,14,14]
    - 前者为拷贝引用，a.expand(3,3,14,14) 新维度[3,3,14,14],仅1可以变,拷贝引用，一个变了一群变
        - 懒得写的话，保持一致的话写-1，比如a.expand(-1,3,-1,-1)
    - Repeat 深拷贝，占用内存，但是是实打实的拷贝，另外，内部的数字为每个数字的拷贝次数
    
### Tensor的转置
- .t 转置函数，仅仅可以对维度为2的搞事情
- Transpose(1,3) 对1,3的维度进行转置，为了保证存储连续性，最好后接contiguous()
- permute（0,2,3,1）一次转置多个维度，内部为新的顺序