import torch
import torch.nn as nn
import torch.optim as optim


"""
我们可以设计一个共享参数的神经网络模型，以便每个原子的神经网络都使用相同的参数。这样可以减少模型的参数数量，并且提高模型的泛化能力。

具体实现
定义环境描述符：为每个原子计算径向描述符和角描述符。
定义共享参数的神经网络：创建一个共享参数的神经网络模型。
组合子神经网络：将各个原子的能量相加以得到总能量。
训练过程：使用均方误差（MSE）作为损失函数，进行训练。


"""

# 环境描述符类
class Descriptor:
    def __init__(self, eta, rs, zeta, lambda_):
        self.eta = eta
        self.rs = rs
        self.zeta = zeta
        self.lambda_ = lambda_

    def calculate_radial_descriptor(self, rij):
        return torch.exp(-self.eta * (rij - self.rs) ** 2)

    def calculate_angular_descriptor(self, rij, rik, theta):
        cos_theta = torch.cos(theta)
        return (2 ** (1 - self.zeta)) * (1 + self.lambda_ * cos_theta) ** self.zeta * \
               torch.exp(-self.eta * (rij ** 2 + rik ** 2 - 2 * rij * rik * cos_theta))

# 共享参数的神经网络类
class SharedNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SharedNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# 总体神经网络类
class TotalNeuralNetwork(nn.Module):
    def __init__(self, num_atoms, input_size, hidden_size, output_size):
        super(TotalNeuralNetwork, self).__init__()
        self.num_atoms = num_atoms
        self.shared_nn = SharedNeuralNetwork(input_size, hidden_size, output_size)

    def forward(self, descriptors):
        total_energy = 0
        for i in range(self.num_atoms):
            atom_energy = self.shared_nn(descriptors[i])
            total_energy += atom_energy
        return total_energy

# 示例数据
def generate_data(num_samples=100):
    # 生成一些随机数据作为示例
    inputs = [torch.rand((num_samples, 5)) for _ in range(3)]
    targets = torch.rand((num_samples, 1))
    return inputs, targets

# 主程序
if __name__ == '__main__':
    # 参数设置
    eta = 0.5
    rs = 1.0
    zeta = 2.0
    lambda_ = -1.0
    input_size = 5
    hidden_size = 10
    output_size = 1
    learning_rate = 0.01
    num_epochs = 500
    num_atoms = 3

    # 初始化描述符
    descriptor = Descriptor(eta, rs, zeta, lambda_)

    # 初始化总体神经网络
    model = TotalNeuralNetwork(num_atoms, input_size, hidden_size, output_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 生成示例数据
    inputs, targets = generate_data()

    # 训练过程
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 打印训练后的参数
    print("\n------------------------------------打印训练后的参数--------------------------------")
    print("Trained Parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param}")
    print("------------------------------------------------------------------------------------\n")

    # 测试预测
    test_input = [torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float32) for _ in range(num_atoms)]
    predicted_energy = model(test_input)
    print(f'\nPredicted Energy: {predicted_energy.item()}')




"""
解释
1.环境描述符：
Descriptor 类定义了径向描述符和角描述符的计算方法。

2.共享参数的神经网络：
SharedNeuralNetwork 类定义了一个简单的多层感知器（MLP），用于预测单个原子的能量。
所有原子共享同一个神经网络实例。

3. 总体神经网络：
TotalNeuralNetwork 类使用一个共享的神经网络实例来预测每个原子的能量，并将它们相加以得到总能量。

4. 生成示例数据：
generate_data 函数生成了三个原子的描述符和目标能量。

5. 打印神经网络参数：
在训练前后分别打印神经网络的参数，确认它们是否一致。

6. 训练过程：
使用均方误差（MSE）作为损失函数。
使用 Adam 优化器进行参数更新。

7. 测试预测：
对一个测试输入进行预测并打印结果。

通过打印神经网络的参数，你可以确认每个原子的神经网络参数是否一致。
在这个示例中，所有原子共享同一个神经网络实例，因此它们的参数是完全一致的。
"""

#TODO: 如何根据元素类型的区分，然后使用不同的神经网络？ 比如H2O体系，H和O的神经网络不同。使用各自的神经网络进行预测。

import torch
import torch.nn as nn
import torch.optim as optim

# 环境描述符类
class Descriptor:
    def __init__(self, eta, rs, zeta, lambda_):
        self.eta = eta
        self.rs = rs
        self.zeta = zeta
        self.lambda_ = lambda_

    def calculate_radial_descriptor(self, rij):
        return torch.exp(-self.eta * (rij - self.rs) ** 2)

    def calculate_angular_descriptor(self, rij, rik, theta):
        cos_theta = torch.cos(theta)
        return (2 ** (1 - self.zeta)) * (1 + self.lambda_ * cos_theta) ** self.zeta * \
               torch.exp(-self.eta * (rij ** 2 + rik ** 2 - 2 * rij * rik * cos_theta))

# 元素特定的神经网络类
class ElementSpecificNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ElementSpecificNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# 总体神经网络类
class TotalNeuralNetwork(nn.Module):
    def __init__(self, num_atoms, element_types, input_size, hidden_size, output_size):
        super(TotalNeuralNetwork, self).__init__()
        self.num_atoms = num_atoms
        self.element_types = element_types
        self.neural_networks = nn.ModuleDict({
            element: ElementSpecificNeuralNetwork(input_size, hidden_size, output_size)
            for element in element_types
        })

    def forward(self, descriptors, element_types):
        total_energy = 0
        for i in range(self.num_atoms):
            element = element_types[i]
            atom_energy = self.neural_networks[element](descriptors[i])
            total_energy += atom_energy
        return total_energy

# 示例数据
def generate_data(num_samples=100):
    # 生成一些随机数据作为示例
    inputs = [torch.rand((num_samples, 5)) for _ in range(3)]
    targets = torch.rand((num_samples, 1))
    element_types = ['H', 'O', 'H']  # 示例元素类型
    return inputs, targets, element_types

# 主程序
if __name__ == '__main__':
    # 参数设置
    eta = 0.5
    rs = 1.0
    zeta = 2.0
    lambda_ = -1.0
    input_size = 5
    hidden_size = 10
    output_size = 1
    learning_rate = 0.01
    num_epochs = 500
    num_atoms = 3
    element_types = ['H', 'O', 'H']

    # 初始化描述符
    descriptor = Descriptor(eta, rs, zeta, lambda_)

    # 初始化总体神经网络
    model = TotalNeuralNetwork(num_atoms, element_types, input_size, hidden_size, output_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 生成示例数据
    inputs, targets, element_types = generate_data()

    # 训练过程
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(inputs, element_types)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 打印训练后的参数
    print("\n------------------------------------打印训练后的参数--------------------------------")
    print("Trained Parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param}")
    print("------------------------------------------------------------------------------------\n")

    # 测试预测
    test_input = [torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float32) for _ in range(num_atoms)]
    predicted_energy = model(test_input, element_types)
    print(f'\nPredicted Energy: {predicted_energy.item()}')