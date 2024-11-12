import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x

def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)

def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data: # 遍历测试数据。x 是输入图像（形状为 (batch_size, 28, 28)），y 是对应的标签（形状为 (batch_size)）。
            outputs = net.forward(x.view(-1, 28 * 28)) # 将图像展平为一个 784 维的向量(batch_size * 748)，然后传入网络进行前向传播，得到每个样本的输出。
            for i, output in enumerate(outputs): # outputs是batch_size个长为10的向量，表示概率
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

def main():

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()

    print("initial accuracy:", evaluate(test_data, net))
    # 优化器是用于更新模型参数的算法。其主要作用是根据损失函数和梯度信息，通过一定的更新规则来调整网络中的权重和偏置，以便模型在训练过程中能够逐渐学习到更好的参数，从而最小化损失函数。
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # Adam（Adaptive Moment Estimation）是一种自适应的优化算法，结合了两个经典的优化算法的优点：
    # 动量（Momentum）：利用历史梯度来加速收敛，减少震荡。
    # 自适应学习率（Adagrad）：为每个参数分配不同的学习率，较大梯度的参数会有较小的学习率，较小梯度的参数会有较大的学习率。
    for epoch in range(5):
        for (x, y) in train_data:
            net.zero_grad()
            output = net.forward(x.view(-1, 28 * 28)) # 前向传播
            loss = torch.nn.functional.nll_loss(output, y) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新模型中的所有可学习参数（如权重和偏置），使其向着最小化损失函数的方向进行调整。
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break # 随机抽四张图，进行预测
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28))) # 预测时取批次第一张图
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()

if __name__ == '__main__':
    main()


