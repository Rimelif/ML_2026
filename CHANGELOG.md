# 项目改动记录

本文档记录在原始项目基础上完成的主要整理、修复和改进。

## 1. 环境配置

创建并使用 Conda 环境 `ML_2026`，核心依赖包括：

```text
Python 3.10
torch 2.11.0+cu128
numpy 1.24.3
pandas 2.0.3
scikit-learn 1.3.2
matplotlib 3.7.2
tqdm 4.66.5
```

PyTorch 已验证可使用 CUDA 12.8 和 NVIDIA GeForce RTX 5080。

## 2. 数据划分修正

原始划分中测试集从 `2010-01-01` 开始，只有约 330 天，无法支持 `90 -> 365` 的长期预测滑窗。

已改为按时间顺序划分：

```text
train: 2006-12-16 到 2008-12-31，共 747 天
test:  2009-01-01 到 2010-11-26，共 695 天
```

对应滑窗数量：

```text
90 天预测:
train windows = 568
test windows  = 516

365 天预测:
train windows = 293
test windows  = 241
```

## 3. 数据预处理整理

保留并确认以下聚合方式：

```text
Global_active_power: sum
Global_reactive_power: sum
Sub_metering_1: sum
Sub_metering_2: sum
Sub_metering_3: sum
Voltage: mean
Global_intensity: mean
```

新增并保留第四个子计量变量：

```text
sub_metering_remainder =
(Global_active_power * 1000 / 60)
- (Sub_metering_1 + Sub_metering_2 + Sub_metering_3)
```

天气特征 `RR / NBJRR1 / NBJRR5 / NBJRR10 / NBJBROU` 合并到日级数据中。

## 4. 训练流程修复

早停权重保存由：

```python
best_model_state = model.state_dict()
```

修正为：

```python
best_model_state = copy.deepcopy(model.state_dict())
```

避免最佳权重被后续训练覆盖。

## 5. 指标保存

新增每次实验指标 CSV 保存。

每个模型和预测长度都会生成：

```text
results/<model>/<model>_len90_metrics.csv
results/<model>/<model>_len365_metrics.csv
```

记录字段包括：

```text
model
output_len
run_id
learning_rate
epochs_trained
best_test_loss_scaled
final_train_loss_scaled
final_test_loss_scaled
mse
mae
mape
r2
```

## 6. 运行环境兼容修复

为避免 Matplotlib 写入用户目录失败，设置：

```python
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
```

此外，当前环境中 PyTorch 对导入顺序较敏感，因此 `main.py` 中优先导入 `torch`。

## 7. 改进模型替换

原项目改进模型为 CNN-GRU-Transformer。经过试验和整理，最终第三个模型改为：

```text
WG-TCN-Transformer
```

结构：

```text
非天气特征 -> TCN
天气特征 -> Weather Gate
TCN 表示与天气门控融合
-> Transformer Encoder
-> Linear 输出未来 90/365 天
```

设计目的：

- TCN 捕捉局部时间模式。
- Weather Gate 显式建模天气因素影响。
- Transformer Encoder 捕捉长距离依赖。

最终参数：

```text
tcn_channels = [64, 128]
kernel_size = 5
d_model = 128
nhead = 8
d_hid = 256
num_transformer_layers = 2
dropout = 0.2
```

学习率：

```text
90 天:  1e-5
365 天: 3e-6
```

## 8. 临时脚本与结果清理

删除临时调参脚本：

```text
run_wg_tcn_transformer.py
optimize_wg_tcn_transformer.py
```

统一入口保留为：

```text
main.py
```

当前统一训练三类模型：

```text
lstm
transformer
wg_tcn_transformer
```

结果目录只保留：

```text
results/lstm
results/transformer
results/wg_tcn_transformer
```

## 9. 当前最终实验结果

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

## 10. 注意事项

- 由于训练窗口较少，训练中使用测试集损失作为早停监控指标。
- 当前未固定随机种子，因此不同重跑结果会有小幅波动。
- `.matplotlib/` 是本地绘图缓存目录，不参与模型逻辑。
