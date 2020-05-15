import torch

x = torch.ones(100, 16, 384, 384) * 666

layer = torch.nn.BatchNorm2d(16)
last = 0
for _ in range(5):
    print("prdct running mean", (0.9*last) + 66.6)
    out = layer(x)
    # print(out[0][0][0][0])
    print("running mean",layer.running_mean)
    last = layer.running_mean
    print("============")

# print(layer.running_mean)
# print(layer.running_var)
#
# print("testing")
# layer.eval()
# out2 = layer(x)
# print(layer.running_mean)
# print(layer.running_var)
# print(layer.bias)
# print(layer.weight)
# print(out2[0][0][0][0])
