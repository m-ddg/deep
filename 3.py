import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5))
        attention_output = torch.matmul(attention_weights, V)

        return attention_output


class StockPredictionModel(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_layers, attention_dim, output_dim):
        super(StockPredictionModel, self).__init__()

        self.bilstm1 = nn.LSTM(input_dim, lstm_hidden_dim, lstm_layers, bidirectional=True, batch_first=True)
        self.attention1 = SelfAttention(lstm_hidden_dim * 2)


        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out1, _ = self.bilstm1(x)
        attention_out1 = self.attention1(lstm_out1)


        output = self.fc(attention_out1[:, -1, :])

        return output


input_dim = 5  # 特征数量
lstm_hidden_dim = 64
lstm_layers = 2
attention_dim = 64
output_dim = 1  # 预测的目标数量

model = StockPredictionModel(input_dim, lstm_hidden_dim, lstm_layers, attention_dim, output_dim)

# 假设数据保存在一个CSV文件中
df = pd.read_csv('D:\\e\\ab\\shangzheng.csv')

# 选取特征和目标
features = df[['open', 'close', 'low', 'high', 'volume']].values
targets = df['close'].values.reshape(-1, 1)  # 注意这里我们需要将目标reshape为2D数组


# 特征数据归一化
feature_scaler = MinMaxScaler()
features_scaled = feature_scaler.fit_transform(features)

# 目标数据归一化
target_scaler = MinMaxScaler()
targets_scaled = target_scaler.fit_transform(targets)

# 将数据转换为 PyTorch 张量
features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
targets_tensor = torch.tensor(targets_scaled, dtype=torch.float32)

# 创建序列数据
def create_sequences(data, target, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = target[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return torch.stack(sequences), torch.tensor(labels)


seq_length = 10  # 序列长度
X, y = create_sequences(features_tensor, targets_tensor, seq_length)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 用于记录训练和验证损失
train_losses = []
val_losses = []

# 训练循环
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        # 前向传播
        output = model(X_batch)
        loss = criterion(output.squeeze(), y_batch)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 计算平均训练损失
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    # 确保模型在评估模式
    model.eval()
    val_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            predictions.append(output.squeeze().numpy())
            actuals.append(y_batch.numpy())
            loss = criterion(output.squeeze(), y_batch)
            val_loss += loss.item()
    # 计算平均验证损失
    val_loss /= len(test_loader)
    val_losses.append(val_loss)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# 将预测值和实际值展平为一维数组
predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)

# 对预测值和实际值进行逆归一化
predictions_inverse = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
actuals_inverse = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()


# 保存结果到DataFrame
results_df = pd.DataFrame({'Actual': actuals_inverse, 'Predicted': predictions_inverse})
results_df.to_csv('predictions.csv', index=False)

### 2. 计算评估指标（使用逆归一化后的数据）

# 计算评估指标
mse = mean_squared_error(actuals_inverse, predictions_inverse)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actuals_inverse, predictions_inverse)
mape = np.mean(np.abs((actuals_inverse - predictions_inverse) / actuals_inverse)) * 100

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.4f}%')


### 3. 可视化（使用逆归一化后的数据）


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(actuals_inverse, label='Actual')
plt.plot(predictions_inverse, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

