import numpy as np
import random
import Chap6_GradientDesend.mnist_loader as mnist_loader


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class MLP_NP:
    def __init__(self, sizes: list):
        """

        :param sizes:[784,30,10]
        """
        self.sizes = sizes
        self.num_layers = len(sizes) - 1
        # sizes[784,30,10]
        # w:[ch_out,ch_in] -> [30,784],[10,30]
        # b:[ch_out] -> [30],[10]
        self.weights = np.array([np.random.randn(chout, chin) for chout, chin in zip(sizes[1:], sizes[:-1])])
        self.bias = np.array([np.random.randn(chout, 1) for chout in sizes[1:]])

    # 预测用的
    def forward(self, x):
        """

        :param x:[784,1]
        :return:[10,1]
        """
        for b, w in zip(self.bias, self.weights):
            # [30*784]@[784,1]=>[30,1]
            z = np.dot(w, x) + b
            # [30,1]=>[30,1]
            x = sigmoid(z)

        return x

    def backprop(self, x, y):
        """
        forward + backward
        :param x: [784]
        :param y:
        :return:
        """
        nabla_w = np.array([np.zeros(w.shape) for w in self.weights])
        nabla_b = np.array([np.zeros(b.shape) for b in self.bias])

        # 每层的激活值 O
        activations = [x]

        # 每层的z(wx+b的那个)
        zs = []
        activation = x
        # forward运算，需要记录每一个z以及o来保存
        for b, w in zip(self.bias, self.weights):
            # print(b.shape)
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            # 存住每一个z和x->那个O
            zs.append(z)
            activations.append(activation)
        # 2. backward
        # 2.1 gradient on output layer Ok*(1-Ok)*(tk-Ok)
        # 输出层:[10,1]，delta:[10,1]
        delta = activations[-1] * (1 - activations[-1]) * (activations[-1] - y)
        nabla_b[-1] = delta
        # delta [10,1],activations[-2]是[30,1],w应该是[10,30]
        # 根据广播原理，应该是[10,1]*[1,30]
        # 但是验证了一下，对于几根柱子来说，点乘与矩阵乘法是等价的
        # nabla_w[-1] = delta * activations[-2].T  # 这里我和他的不一样，他的是矩阵乘法，我是点乘，实际上一样
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # 2.2 compute hidden layer
        # weights的定义是根据输出层决定的
        for l in range(2, self.num_layers + 1):  # [-2,-3....-总layer]
            l = -l
            z = zs[l]
            a = activations[l]  # [30,1]
            delta = np.dot(self.weights[l + 1].transpose(), delta) * a * (1 - a)
            nabla_w[l] = np.dot(delta, activations[l - 1].T)
            nabla_b[l] = delta

        return nabla_w, nabla_b

    def train(self, trainging_data, epochs, batchsz, lr, test_data):
        """

        :param trainging_data: list of (x,y)
        :param epochs: 1000
        :param batchsz: 10
        :param lr: 0.01
        :param test_data: list of (x,y)
        :return:
        """
        if test_data:
            n_test = len(test_data)
        n = len(trainging_data)
        for j in range(epochs):
            random.shuffle(trainging_data)
            mini_bacthes = [trainging_data[k:k + batchsz] for k in range(0, n, batchsz)]
            for mini_bacth in mini_bacthes:
                self.update_mini_bacth(mini_bacth, lr)
            if test_data:
                print("Epoch{0}:{1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch{0} complete".format(j))

    def update_mini_bacth(self, batch, lr):
        """

        :param batch:
        :param lr:
        :return:
        """
        nabla_w = np.array([np.zeros(w.shape) for w in self.weights])
        nabla_b = np.array([np.zeros(b.shape) for b in self.bias])

        # 对每一个batch
        for x, y in batch:
            nabla_w_, nabla_b_ = self.backprop(x, y)
            nabla_w += nabla_w_
            nabla_b += nabla_b_
        nabla_w = nabla_w / len(batch)
        nabla_b = nabla_b / len(batch)
        # print(nabla_w.shape)
        # print(self.weights.shape)
        # w = w-lr*nw
        self.weights -= nabla_w * lr
        self.bias -= nabla_b * lr

    def evaluate(self, test_data):
        """

        :param test_data: list of (x,y)
        :return:
        """

        res = np.array([(np.argmax(self.forward(x)), y) for x, y in test_data])
        correct = sum([int(pred == y) for pred, y in res])

        return correct / len(res)


def main():
    training_data, validation_data, testdata = mnist_loader.load_data_wrapper()
    net = MLP_NP([784, 30, 10])
    import time
    st = time.time()
    net.train(training_data, epochs=10, batchsz=25, lr=0.03, test_data=testdata)
    print("{0} epochs, total time spent: {1}s".format(10, time.time() - st))



if __name__ == '__main__':
    main()
