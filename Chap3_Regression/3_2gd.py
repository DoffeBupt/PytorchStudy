import numpy as np


# loss = (Wx+b-y)^2

# 计算误差和
def compute_error(b, w, points):
    totalError = 0;
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (w * x + b - y) ** 2
    return totalError / float(len(points))


# 算梯度
# loss = (Wx+b-y)^2

def step_gradient(b_current, w_current, points, lr):
    b_gradient = 0;
    w_gradient = 0;
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += 2 / N * (w_current * x + b_current - y)
        w_gradient += 2 / N * x * (w_current * x + b_current - y)
    new_b_current = b_current - lr * b_gradient
    new_w_current = w_current - lr * w_gradient
    return new_b_current, new_w_current


# 迭代优化
def gd_runner(points, starting_b, starting_w, lr, num_iter):
    b = starting_b
    w = starting_w
    for i in range(num_iter):
        b, w = step_gradient(b, w, points, lr)
    return b, w


# 主逻辑
def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    lr = 0.0001
    init_b = 0
    init_w = 0
    num_iter = 10000
    print("starting gd at b = {0}, w = {1}, error = {2}".format(init_b, init_w, compute_error(init_b, init_w, points)))
    print("Running...")
    b, w = gd_runner(points, init_b, init_w, lr, num_iter)
    print("Finished, result in b = {0}, w = {1}, error = {2}".format(b, w, compute_error(b, w, points)))


if __name__ == '__main__':
    run()
