import torch


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def himmelblau2(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


x = torch.tensor([4., 0.], requires_grad=True)
x1 = torch.tensor([4.], requires_grad=True)
x2 = torch.tensor([0.], requires_grad=True)
# x = x.cuda()
optimzer = torch.optim.Adam([x1, x2], lr=1e-3)
for step in range(20000):
    pred = himmelblau2(x1, x2)
    optimzer.zero_grad()
    pred.backward()
    optimzer.step()
    if step % 2000 == 0:
        print('step{}:x={},f(x)={}'.format(step, x1.tolist(), pred.item()))
