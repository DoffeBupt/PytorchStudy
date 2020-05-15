import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from Chap10_ResNet.LeNet5 import LeNet5
from Chap10_ResNet.resnet import ResNet18
from torch import nn, optim


def main():
    batchsz = 32
    # 目录文件夹名字,是不是训练的,数据增强,是否下载
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = iter(cifar_train).__next__()  # 得到了一个batch的数据
    print("x:", x.shape, "  label:", label.shape)

    device = torch.device('cuda')
    model = ResNet18().to(device)
    print(model)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=(1e-3))
    for epoch in range(1000):
        model.train()

        for batchidx, (x, label) in enumerate(cifar_train):
            # x:     [b,3,32,32]
            # lable: [b,10]
            x, label = x.to(device), label.to(device)
            logits = model(x)
            # [b,10] label:[10]
            loss = criteon(logits, label)
            # backprop
            # 在每次backprop的时候,optimizer的梯度会累加
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, loss.item())  # 这里的loss其实是最后一个batch的loss
        # test的时候,为了防止他乱七八糟去算梯度,这样会保险一些
        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)
                # [b,10]->[b]
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num
            print(epoch, acc)


if __name__ == '__main__':
    main()
