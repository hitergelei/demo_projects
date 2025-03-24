import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 步骤 2: 定义ADP势函数
# 定义一个简单的ADP势函数，这里简化为一个二维版本。
def adp_potential(r_ij, theta_ijk, params):
    A, alpha, r0, B, n, C = params
    term1 = A * torch.exp(-alpha * (r_ij - r0)**2)
    term2 = B * (r_ij / r0)**n
    term3 = C * torch.cos(theta_ijk)
    return term1 + term2 + term3


# 步骤 3: 定义残差网络模型
# 我们将使用一个简单的残差网络来拟合ADP势的参数。
class ResidualADPModel(nn.Module):
    def __init__(self):
        super(ResidualADPModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # 输入维度为2（r_ij 和 theta_ijk）
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 6)  # 输出维度为6（ADP势的参数）
        self.relu = nn.ReLU()
        self.res_block = ResidualBlock(64, 64)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.res_block(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    

# 步骤 4: 定义残差块 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out


# 步骤 5: 准备数据集
# 假设我们有一些训练数据，包括距离 ( r_{ij} ) 和角度 ( \theta_{ijk} )，以及对应的能量值。

# 生成一些随机数据作为示例
r_ij = np.random.rand(100)  # 100个样本的距离
theta_ijk = np.random.rand(100) * np.pi  # 100个样本的角度
energy_true = np.random.rand(100)  # 100个样本的真实能量值

# 转换为Tensor
r_ij_tensor = torch.tensor(r_ij, dtype=torch.float32).unsqueeze(1)
theta_ijk_tensor = torch.tensor(theta_ijk, dtype=torch.float32).unsqueeze(1)
energy_true_tensor = torch.tensor(energy_true, dtype=torch.float32).unsqueeze(1)

# 合并输入数据
input_data = torch.cat((r_ij_tensor, theta_ijk_tensor), dim=1)

# 创建数据加载器
train_dataset = TensorDataset(input_data, energy_true_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)




# 步骤 6: 训练模型
model = ResidualADPModel()
criterion = nn.MSELoss()  # 使用均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for inputs, energy_true in train_loader:
        optimizer.zero_grad()
        predicted_params = model(inputs)
        predicted_energy = []
        for i in range(len(inputs)):
            param = predicted_params[i]
            r_ij_i = inputs[i, 0]
            theta_ijk_i = inputs[i, 1]
            energy_i = adp_potential(r_ij_i, theta_ijk_i, param)
            predicted_energy.append(energy_i)
        predicted_energy = torch.stack(predicted_energy)
        loss = criterion(predicted_energy, energy_true)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    



import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.manifold import TSNE

# 假设X是高维数据，P是对应的属性值
X = np.random.random((100, 10))  # 100个样本，每个样本10个特征
P = np.random.random(100)  # 100个属性值

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)

# 定义网格范围
x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()

# 创建网格
grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

# 插值
points = X_2d
values = P
grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

# 绘制热力图
plt.figure(figsize=(10, 6))
plt.imshow(grid_z.T, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
plt.colorbar(label='Density')  # 显示颜色条
plt.scatter(X_2d[:, 0], X_2d[:, 1], c='white', edgecolors='black', s=50)  # 显示原始点
plt.title('t-SNE with Density Heatmap')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.savefig('t-SNE_heatmap-2D密度图.jpg')
plt.show()





import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

# 假设X是高维数据，P是对应的属性值
X = np.random.random((100, 10))  # 100个样本，每个样本10个特征
P = np.random.random(100)  # 100个属性值

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)

# 为二维空间中的每个点估计属性值  RNF方法
def radial_neighbor_regression(X_2d, X_high_dim, P, n_neighbors=5):
    # 计算高维空间中的距离
    distances = cdist(X_high_dim, X_high_dim, metric='euclidean')

    # 为每个点找到最近的n_neighbors个邻居的索引
    indices = np.argsort(distances, axis=1)[:, 1:n_neighbors+1]

    # 获取这些邻居的距离
    row_indices = np.arange(len(X_high_dim)).reshape(-1, 1)  # 转换为二维数组以便与indices匹配
    distances = distances[row_indices, indices]

    # 确保没有距离为0（自己到自己的距离）
    distances[distances == 0] = np.finfo(float).eps

    # 计算权重，使用距离的倒数
    weights = 1 / distances
    weights /= np.sum(weights, axis=1, keepdims=True)

    # 获取邻居的属性值
    P_neighbors = P[indices]

    # 计算加权属性值
    P_2d = np.sum(weights * P_neighbors, axis=1)
    return P_2d

# 调用函数
P_2d = radial_neighbor_regression(X_2d, X, P)

# 定义网格范围
x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()

# 创建网格
grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

# 插值
points = X_2d
values = P_2d
grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

# 绘制热力图
plt.figure(figsize=(10, 6))
plt.imshow(grid_z.T, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
plt.colorbar(label='Attribute Value')  # 显示颜色条
plt.scatter(X_2d[:, 0], X_2d[:, 1], c='white', edgecolors='black', s=50)  # 显示原始点
plt.title('t-SNE with RNR Heatmap')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.savefig('t-SNE_RNR_heatmap——2.jpg')
plt.show()