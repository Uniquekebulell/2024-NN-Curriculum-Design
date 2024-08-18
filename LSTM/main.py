import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import time
import math
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# 1.导入数据
# filepath = '/root/LSTM-Project/rlData.csv'  # 导入数据
filepath = 'F:/Interests/Postgraduate Exam/Project/2024 NN Curriculum Design/LSTM/rlData.csv'  # 导入数据
data = pd.read_csv(filepath)
data = data.sort_values('Date')  # 将数据按照日期顺序进行排序
print(data.head())  # 打印前几条数据
print(data.shape)   # 打印维度
print("----------------------------------------1.导入数据完成--------------------------------------")

# # 2.将股票数据收盘价(Label)进行可视化展示
# sns.set_style("darkgrid")   # 用sns库定义风格，即带有网格线的黑色背景
# plt.figure(figsize=(15, 9))  # 创建一个窗口，设置图像大小为15*9英寸
# plt.plot(data[['Label']])    # 绘制Label列的折线图
# plt.xticks(range(0, data.shape[0], 20), data['Date'].loc[::20], rotation=45) # 设置横坐标，每隔20个显示一个日期，并且倾斜45度避免显示重叠
# plt.title("****** Stock Price", fontsize=18, fontweight='bold')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Label Price (USD)', fontsize=18)
# plt.show()
print("----------------------------------------2.可视化展示股票数据收盘价(Label)完成--------------------------------------")

# 3.特征工程
# price = data[['Label']]  # 选取Label作为特征
price = data[['Label']]  # 选取Label作为特征
print(price.info())  # info()函数在pandas库中用于输出有关DataFrame或Series对象的详细信息
# 进行不同的数据缩放，将数据缩放到 -1和 1之间
# scaler = MinMaxScaler(feature_range=(-1, 1))  # 设置特征范围是[-1,1]
# price['Label'] = scaler.fit_transform(price['Label'].values.reshape(-1, 1))  # 将Label列的数据缩放到-1,1之间
# print(price['Label'].shape)
scaler = MinMaxScaler(feature_range=(-1, 1))
price_c = price.copy()  # 先建立副本，不直接在原始数据上进行操作
price_c['Label'] = scaler.fit_transform(price_c['Label'].values.reshape(-1, 1))  # 在副本上进行操作
price = price_c.copy() # 还原
print(price['Label'].shape)
print("----------------------------------------3.特征选取完毕--------------------------------------")

# 4.数据集制作
# 今天的收盘价预测明天的收盘价
def split_data(stock, lookback):  # stock参数表示 Label即选取的特征这一列，lookback表示观察的跨度
    data_raw = stock.to_numpy()  # 将 stock转化为 ndarray类型
    data = []
    # print(data)

    # you can free play（seq_length）
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data)
    test_set_size = int(np.round(0.01 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-1, :]  # 将data的前train_set_size行，去掉最后一列的数据，赋值给x_train
    y_train = data[:train_set_size, -1, :]   # 将data的前train_set_size行，最后一列的数据，赋值给y_train

    x_test = data[train_set_size:, :-1, :]  # 将data从train_set_size开始到最后一行，去掉最后一列的数据，赋值给x_test
    y_test = data[train_set_size:, -1, :]  #将data从train_set_size开始到最后一行，最后一列的数据，赋值给y_test

    return [x_train, y_train, x_test, y_test]

lookback = 20
x_train, y_train, x_test, y_test = split_data(price, lookback)
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)
print("----------------------------------------4.数据集制作完成--------------------------------------")

# 注意：pytorch的nn.LSTM input shape=(seq_length, batch_size, input_size)
# 5.模型构建 —— LSTM
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)
# 输入的维度为1，只有Label收盘价
input_dim = 1
# 隐藏层特征的维度
hidden_dim = 32
# 循环的layers
num_layers = 2
# 预测后一天的收盘价
output_dim = 1
num_epochs = 100

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()  # h0表示细胞初始隐藏状态，需要计算梯度
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()  # c0表示初始细胞状态，与h0的形状都是(num_layers, batch_size, hidden_dim)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
print("----------------------------------------5.LSTM模型构建成功--------------------------------------")

# 6.模型训练
hist = np.zeros(num_epochs)   # 创建一个长度为 num_epochs的全零数组，用于存储每个epoch的损失值
start_time = time.time()
lstm = []

for t in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_lstm)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

training_time = time.time() - start_time
print("Training time: {}".format(training_time))

predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
print("predict data:\n")
print(predict)
original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))
print("original data:\n")
print(original)
print("----------------------------------------6.模型训练成功--------------------------------------")

# 7.模型结果可视化
sns.set_style("darkgrid")

fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)  # 子图之间的间距

plt.subplot(1, 2, 1)
ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (LSTM)", color='tomato')
print(predict.index)
print(predict[0])
ax.set_title('Stock price(train/test = 99:1)', size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Cost (USD)", size = 14)
ax.set_xticklabels('', size=10)   # 不显示横坐标数值

plt.subplot(1, 2, 2)
ax = sns.lineplot(data=hist, color='royalblue')
ax.set_xlabel("Epoch", size = 14)
ax.set_ylabel("Loss", size = 14)
ax.set_title("Training Loss(train/test = 99:1)", size = 14, fontweight='bold')
fig.set_figheight(6)
fig.set_figwidth(16)
plt.show()
print("----------------------------------------7.模型结果可视化成功--------------------------------------")

# 8.模型验证
# print(x_test[-1])
# make predictions
y_test_pred = model(x_test)

# invert predictions  反归一化操作。训练阶段需要先归一化，以便更好地训练模型；而预测阶段则需要将数据还原到原来的数值范围内，即进行反归一化操作。
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test_lstm.detach().numpy())

# calculate root mean squared error 计算均方根误差
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))  # mean_squared_error()用来计算均方误差
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
lstm.append(trainScore)
lstm.append(testScore)
lstm.append(training_time)

# shift train predictions for plotting  创建一个与原始价格数据相同形状的空数组，并将训练集预测结果填充到该数组中。
trainPredictPlot = np.empty_like(price)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

# shift test predictions for plotting
testPredictPlot = np.empty_like(price)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred)+lookback-1:len(price)-1, :] = y_test_pred

original = scaler.inverse_transform(price['Label'].values.reshape(-1,1))

predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
predictions = np.append(predictions, original, axis=1)
result = pd.DataFrame(predictions)

fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                    mode='lines',
                    name='Train prediction')))
fig.add_trace(go.Scatter(x=result.index, y=result[1],
                    mode='lines',
                    name='Test prediction'))
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                    mode='lines',
                    name='Actual Value')))
fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        # linecolor='white',
        linecolor='black',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Label (USD)',
        titlefont=dict(
            family='Rockwell',
            size=12,
            # color='white',
            color='black',
        ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        # linecolor='white',
        linecolor='black',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            # color='white',
            color='black',
        ),
    ),
    showlegend=True,
    # template = 'plotly_dark'
    template = 'plotly'


)

annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Results (LSTM) (train/test = 99:1)',
                              font=dict(family='Rockwell',
                                        size=26,
                                        # color='white'),
                                        color='black'),
                        showarrow=False))
fig.update_layout(annotations=annotations)
fig.show()
print("----------------------------------------8.模型验证--------------------------------------")