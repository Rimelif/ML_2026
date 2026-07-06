import torch

import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

def scale_features(train_df, test_df, features):
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    # 存放特征缩放器 
    scalers = {}
    
    # 标准化 Global_active_power（目标值）
    scalers['target'] = StandardScaler()
    train_scaled['Global_active_power'] = scalers['target'].fit_transform(train_df[['Global_active_power']])
    test_scaled['Global_active_power'] = scalers['target'].transform(test_df[['Global_active_power']])

    # 最大-最小缩放特征
    scalers['features'] = MinMaxScaler()
    train_scaled[features] = scalers['features'].fit_transform(train_df[features])
    test_scaled[features] = scalers['features'].transform(test_df[features])

    return train_scaled, test_scaled, scalers

def slide_window(data, input_len = 90, output_len = 90, target_col_index=0):
        """
            初始化数据集。
            
            参数:
            daily_csv_path (str): 日度数据CSV文件路径。
            input_len (int): 输入序列的长度（例如：90天）。
            output_len (int): 预测序列的长度（例如：90天或365天）。
            target_col_index (str): 目标预测列的索引。
        """
        length = len(data) - input_len - output_len + 1
        data_tensor = []
        for i in range(length):
            input_end = i + input_len
            output_end = input_end + output_len
            input = torch.tensor(data[i : input_end, : ], dtype = torch.float32)
            output= torch.tensor(data[input_end : output_end, target_col_index], dtype = torch.float32)
            data_tensor.append((input,output))

        return data_tensor

def plot_pre(save_dir, preds, targets, model_name, output_len, run_id):
    """
    绘制预测值与真实值的对比图，并保存。
    
    参数:
    preds (np.array): 预测值数组, shape: (num_samples, pred_len)
    targets (np.array): 真实值数组, shape: (num_samples, pred_len)
    output_len (int): 预测长度。
    model_name (str): 模型名称。
    run_id (str): 实验轮次ID。
    save_dir (str): 图片保存目录。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(15, 7))

    # 绘制真实值曲线
    plt.plot(np.arange(output_len), targets[0, :], label='Ground Truth')

    # 绘制预测值曲线
    plt.plot(np.arange(output_len), preds[0, :], label='Prediction')
    
    plt.title(f'Prediction vs Ground Truth for {model_name.upper()} ({output_len}-day forecast)')
    plt.xlabel('Time (Days into the future)')
    plt.ylabel('Global Active Power (Daily Sum)')
    plt.legend()
    plt.grid(True)
    
    filename = f"{save_dir}/{model_name}_len{output_len}_run{run_id}.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()

def plot_loss(save_dir, run_id, train_losses, test_losses, model_name, output_len):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss (as Validation)') # 修改图例
    plt.title(f'Training & Test Loss for Run {run_id}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    loss_curve_filename = f"{save_dir}/{model_name}_len{output_len}_loss_curve_run{run_id}.png"
    plt.savefig(loss_curve_filename)
    print(f"Loss curve saved to {loss_curve_filename}")
    plt.close()
