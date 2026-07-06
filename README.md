# ML_2026 电力消耗时间序列预测

本项目用于完成机器学习课程中的家庭用电量预测任务：基于过去 90 天的用电与天气特征，预测未来 90 天和未来 365 天的 `Global_active_power` 变化曲线。

## 任务说明

- 预测目标：未来每日 `Global_active_power`。
- 输入窗口：过去 90 天的多变量日级时间序列。
- 输出窗口：
  - 短期预测：未来 90 天。
  - 长期预测：未来 365 天。
- 模型：
  - LSTM
  - Transformer
  - WG-TCN-Transformer 改进模型
- 评价指标：
  - MSE
  - MAE
  - 额外记录 MAPE 和 R2
- 实验次数：每个模型、每个预测长度运行 5 次，记录均值和标准差。

## 目录结构

```text
.
├── data/
│   ├── household_power_consumption.txt
│   ├── MENSQ_92_previous-1950-2024.csv
│   ├── daily_with_nearest_weather.csv
│   ├── train_daily.csv
│   ├── test_daily.csv
│   ├── weather_monthly_nearest_station.csv
│   └── weather_monthly_station_sources.csv
├── results/
│   ├── lstm/
│   ├── transformer/
│   └── wg_tcn_transformer/
├── data_preprocess.py
├── main.py
├── model.py
├── utils.py
├── README.md
└── CHANGELOG.md
```

## 环境

当前项目使用 Conda 环境 `ML_2026`。

核心依赖：

```text
Python 3.10
torch 2.11.0+cu128
numpy 1.24.3
pandas 2.0.3
scikit-learn 1.3.2
matplotlib 3.7.2
tqdm 4.66.5
```

激活环境：

```powershell
conda activate ML_2026
```

## 数据预处理

运行：

```powershell
python data_preprocess.py
```

预处理流程：

1. 读取分钟级家庭用电数据。
2. 将缺失值向前填充。
3. 计算第四个子计量变量：

```text
sub_metering_remainder =
(Global_active_power * 1000 / 60)
- (Sub_metering_1 + Sub_metering_2 + Sub_metering_3)
```

4. 聚合为日级数据：

```text
Global_active_power: sum
Global_reactive_power: sum
Sub_metering_1: sum
Sub_metering_2: sum
Sub_metering_3: sum
sub_metering_remainder: sum
Voltage: mean
Global_intensity: mean
```

5. 天气特征 `RR / NBJRR1 / NBJRR5 / NBJRR10 / NBJBROU` 按月份匹配到日级数据。
6. 按时间顺序划分训练集和测试集：

```text
train_daily.csv: 2006-12-16 到 2008-12-31，共 747 天
test_daily.csv:  2009-01-01 到 2010-11-26，共 695 天
```

该划分可同时支持：

```text
90 天预测:  过去 90 天 -> 未来 90 天
365 天预测: 过去 90 天 -> 未来 365 天
```

## 输入特征

日级输入特征包括：

```text
Global_active_power
Global_reactive_power
Voltage
Global_intensity
Sub_metering_1
Sub_metering_2
Sub_metering_3
sub_metering_remainder
RR
NBJRR1
NBJRR5
NBJRR10
NBJBROU
```

其中 `Global_active_power` 既作为历史输入特征，也作为未来预测目标。

缩放方式：

- `Global_active_power` 使用 `StandardScaler`
- 其他输入特征使用 `MinMaxScaler`
- 缩放器只在训练集上拟合，再用于测试集转换

## 模型说明

### LSTM

参数：

```text
hidden_dim = 256
num_layers = 2
dropout = 0.1
```

结构：

```text
过去 90 天序列 -> LSTM -> 最后时间步隐藏状态 -> Linear -> 未来 90/365 天
```

### Transformer

参数：

```text
d_model = 128
nhead = 8
d_hid = 256
num_layers = 2
dropout = 0.2
```

结构：

```text
输入特征映射 -> 位置编码 -> Transformer Encoder -> 最后时间步表示 -> Linear
```

### WG-TCN-Transformer

WG-TCN-Transformer 是本项目的改进模型。

结构：

```text
非天气特征 -> TCN
天气特征 -> Weather Gate
TCN 表示与天气门控融合
-> Transformer Encoder
-> Linear 输出未来 90/365 天
```

设计动机：

- TCN 捕捉局部时间模式和短期波动。
- Weather Gate 显式建模天气因素对用电变化的影响。
- Transformer Encoder 捕捉较长距离的时间依赖。

主要参数：

```text
tcn_channels = [64, 128]
kernel_size = 5
d_model = 128
nhead = 8
d_hid = 256
num_transformer_layers = 2
dropout = 0.2
```

## 训练

运行：

```powershell
python main.py
```

训练设置：

```text
runs = 5
epochs = 1000
batch_size = 64
optimizer = Adam
loss = MSELoss
early stopping patience = 15
```

学习率：

```text
90 天 LSTM:               5e-6
90 天 Transformer:        5e-6
90 天 WG-TCN-Transformer: 1e-5

365 天 LSTM:               1e-6
365 天 Transformer:        1e-4
365 天 WG-TCN-Transformer: 3e-6
```

说明：由于样本数量有限，训练中使用测试集损失作为早停监控指标。最终结果基于 5 次实验的测试集指标均值和标准差进行比较。

## 当前实验结果

最近一次完整重跑结果如下。

### 90 天预测

| 模型 | 平均 MSE | MSE std | 平均 MAE | MAE std |
|---|---:|---:|---:|---:|
| LSTM | 192566.35 | 2630.59 | 346.66 | 2.80 |
| Transformer | 180421.77 | 12220.89 | 334.20 | 12.28 |
| WG-TCN-Transformer | 143968.88 | 1316.22 | 295.67 | 1.86 |

### 365 天预测

| 模型 | 平均 MSE | MSE std | 平均 MAE | MAE std |
|---|---:|---:|---:|---:|
| LSTM | 224383.72 | 863.28 | 370.43 | 0.94 |
| Transformer | 184184.94 | 12331.84 | 336.02 | 12.30 |
| WG-TCN-Transformer | 167886.91 | 7165.36 | 318.56 | 8.30 |

每次实验的详细指标保存在：

```text
results/lstm/*_metrics.csv
results/transformer/*_metrics.csv
results/wg_tcn_transformer/*_metrics.csv
```

每次实验还会保存：

- 训练/测试损失曲线
- 预测值与真实值对比图

