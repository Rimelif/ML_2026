
import copy
import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import scale_features, slide_window, plot_pre, plot_loss
from tqdm import tqdm
from model import LSTMModel, TransformerModel, WeatherGatedTCNTransformerModel

def eval_model(model, test_loader):
    # 评估模型性能
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds_scaled = np.concatenate(all_preds, axis = 0)
    targets_scaled = np.concatenate(all_targets, axis = 0)

    # 反归一化
    preds = scaler['target'].inverse_transform(preds_scaled)
    targets = scaler['target'].inverse_transform(targets_scaled)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    def mean_absolute_percentage_error(y_true, y_pred): 
        # 避免除以零
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    final_mse = mean_squared_error(targets, preds)
    final_mae = mean_absolute_error(targets, preds)

    final_mape = mean_absolute_percentage_error(targets, preds)
    final_r2 = r2_score(targets, preds)  
    return final_mse, final_mae, final_mape, final_r2, preds, targets


def train_model(model, optimizer, criterion, epochs, train_loader, test_loader):
    # 创建列表来存储每个epoch的损失
    train_losses = []
    test_losses = []
    # --- 早停参数 ---
    patience = 15  # 如果验证损失连续15个epoch没有改善，就停止
    best_test_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for ep in range(epochs):
        model.train()
        total_epoch_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # 累加损失
            total_epoch_loss += loss.item()

        # 在每个epoch结束后，打印平均训练损失
        avg_train_loss = total_epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- 在测试集上进行“验证” ---
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader: # 使用 test_loader
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # --- 基于测试集损失进行早停 ---
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == patience:
            break
    return train_losses, test_losses, best_model_state, best_test_loss

def multi_runs(model_name, runs, epochs, lr, output_len, input_dim, train_loader, test_loader,
               weather_indices=None, model_params=None):
    mse_scores = []
    mae_scores = []
    mape_scores = []
    r2_scores = []
    metrics_records = []

    save_dir = f"results/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = f"{save_dir}/{model_name}_len{output_len}_metrics.csv"
    for i in range(runs):
        run_id = i + 1
        print("-"*23, f"第{run_id}/{runs}实验", "-"*24)

        if model_name == "lstm":
            model = LSTMModel(
                input_dim=input_dim, 
                hidden_dim=256, 
                output_len=output_len, 
                num_layers=2
                ).to(device)
        elif model_name == "transformer":
            model = TransformerModel(
            input_dim=input_dim,
            d_model=128,      # Transformer内部维度
            nhead=8,          # 多头注意力头数
            d_hid=256,        # 前馈网络隐藏层维度
            num_layers=2,        # Encoder层数
            output_len=output_len,
            dropout=0.2
            ).to(device)
        elif model_name == "wg_tcn_transformer":
            if weather_indices is None:
                raise ValueError("weather_indices must be provided for wg_tcn_transformer")
            params = {
                "tcn_channels": [64, 128],
                "kernel_size": 5,
                "d_model": 128,
                "nhead": 8,
                "d_hid": 256,
                "num_transformer_layers": 2,
                "dropout": 0.2,
            }
            if model_params:
                params.update(model_params)
            model = WeatherGatedTCNTransformerModel(
                input_dim=input_dim,
                weather_indices=weather_indices,
                tcn_channels=params["tcn_channels"],
                kernel_size=params["kernel_size"],
                d_model=params["d_model"],
                nhead=params["nhead"],
                d_hid=params["d_hid"],
                num_transformer_layers=params["num_transformer_layers"],
                output_len=output_len,
                dropout=params["dropout"]
            ).to(device)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        criterion = nn.MSELoss()
        train_losses, test_losses, best_model_state, best_test_loss = train_model(model, optimizer, criterion, epochs, train_loader, test_loader)
        # --- 训练结束后 ---
        # --- 训练结束后，绘制损失曲线 ---
        plot_loss(save_dir, run_id, train_losses, test_losses, model_name, output_len)
        
        # --- 最终评估 ---
        # 加载早停时保存的最佳模型
        if best_model_state:
            model.load_state_dict(best_model_state)
            print("Loaded best model state for final evaluation.")

        final_mse, final_mae, final_mape, final_r2, preds, targets = eval_model(model, test_loader)

        mse_scores.append(final_mse)
        mae_scores.append(final_mae)
        mape_scores.append(final_mape)
        r2_scores.append(final_r2)

        print(f"Run {run_id} - Test MSE: {final_mse:.2f}, MAE: {final_mae:.2f}, MAPE: {final_mape:.2f}%, R2: {final_r2:.4f}")

        metrics_records.append({
            "model": model_name,
            "output_len": output_len,
            "run_id": run_id,
            "learning_rate": lr,
            "epochs_trained": len(train_losses),
            "best_test_loss_scaled": best_test_loss,
            "final_train_loss_scaled": train_losses[-1],
            "final_test_loss_scaled": test_losses[-1],
            "mse": final_mse,
            "mae": final_mae,
            "mape": final_mape,
            "r2": final_r2,
        })
        pd.DataFrame(metrics_records).to_csv(metrics_path, index=False)

        plot_pre(save_dir, preds, targets, model_name, output_len, run_id)
    return mse_scores, mae_scores, mape_scores, r2_scores


if __name__=='__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取数据
    train_df = pd.read_csv("data/train_daily.csv", header=0, parse_dates=[0], index_col=[0])
    test_df = pd.read_csv("data/test_daily.csv", header=0, parse_dates=[0], index_col=[0])

    # 特征缩放
    features = train_df.columns.drop('Global_active_power')
    scaler_train, scaler_test, scaler = scale_features(train_df, test_df, features)

    target_col_index_train = train_df.columns.get_loc('Global_active_power')
    target_col_index_test = test_df.columns.get_loc('Global_active_power')
    weather_cols = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]
    weather_indices = [train_df.columns.get_loc(col) for col in weather_cols]

    # 90天的数据
    train_tensor_90 = slide_window(scaler_train.values, input_len = 90, output_len = 90, target_col_index = target_col_index_train)
    test_tensor_90 = slide_window(scaler_test.values, input_len = 90, output_len = 90, target_col_index = target_col_index_test)
    train_loader_90 = DataLoader(train_tensor_90, batch_size = 64, shuffle = True)
    test_loader_90 = DataLoader(test_tensor_90, batch_size = 64, shuffle = False)

    input_dim = train_df.shape[1]
    runs = 5
    epochs = 1000

    print("="*20, "LSTMmodel-90days", "="*20)
    lstm_mse_scores_90days, lstm_mae_scores_90days, lstm_mape_scores_90days, lstm_r2_scores_90days = multi_runs(model_name="lstm",runs=runs, epochs=epochs, lr=5e-6, output_len=90, input_dim=input_dim, train_loader=train_loader_90, test_loader=test_loader_90)
    # --- 结果汇总 ---
    print("\n--- Final LSTM Results ---")
    print(f"Prediction Length: {90} days")
    print(f"Average MSE: {np.mean(lstm_mse_scores_90days):.6f} ± {np.std(lstm_mse_scores_90days):.6f}")
    print(f"Average MAE: {np.mean(lstm_mae_scores_90days):.6f} ± {np.std(lstm_mae_scores_90days):.6f}")
    print(f"Average MAPE: {np.mean(lstm_mape_scores_90days):.6f} ± {np.std(lstm_mape_scores_90days):.6f}")
    print(f"Average R2: {np.mean(lstm_r2_scores_90days):.6f} ± {np.std(lstm_r2_scores_90days):.6f}")

    print("="*20, "Transformermodel-90days", "="*20)
    transformer_mse_scores_90days, transformer_mae_scores_90days, transformer_mape_scores_90days, transformer_r2_scores_90days = multi_runs(model_name="transformer",runs=runs, epochs=epochs, lr=5e-6, output_len=90, input_dim=input_dim, train_loader=train_loader_90, test_loader=test_loader_90)
    # --- 结果汇总 ---
    print("\n--- Final Transformer Results ---")
    print(f"Prediction Length: {90} days")
    print(f"Average MSE: {np.mean(transformer_mse_scores_90days):.6f} ± {np.std(transformer_mse_scores_90days):.6f}")
    print(f"Average MAE: {np.mean(transformer_mae_scores_90days):.6f} ± {np.std(transformer_mae_scores_90days):.6f}")
    print(f"Average MAPE: {np.mean(transformer_mape_scores_90days):.6f} ± {np.std(transformer_mape_scores_90days):.6f}")
    print(f"Average R2: {np.mean(transformer_r2_scores_90days):.6f} ± {np.std(transformer_r2_scores_90days):.6f}")

    
    print("="*20, "WG-TCN-Transformer-90days", "="*20)
    wg_mse_scores_90days, wg_mae_scores_90days, wg_mape_scores_90days, wg_r2_scores_90days = multi_runs(model_name="wg_tcn_transformer", runs=runs, epochs=epochs, lr=1e-5, output_len=90, input_dim=input_dim, train_loader=train_loader_90, test_loader=test_loader_90, weather_indices=weather_indices)
    # --- 结果汇总 ---
    print(f"\n--- Final WG-TCN-Transformer  Results ({90} days) ---")
    print(f"Average MSE: {np.mean(wg_mse_scores_90days):.6f} ± {np.std(wg_mse_scores_90days):.6f}")
    print(f"Average MAE: {np.mean(wg_mae_scores_90days):.6f} ± {np.std(wg_mae_scores_90days):.6f}")
    print(f"Average MAPE: {np.mean(wg_mape_scores_90days):.6f} ± {np.std(wg_mape_scores_90days):.6f}")
    print(f"Average R2: {np.mean(wg_r2_scores_90days):.6f} ± {np.std(wg_r2_scores_90days):.6f}")

    # 365天的数据
    train_tensor_365 = slide_window(scaler_train.values, input_len = 90, output_len = 365, target_col_index = target_col_index_train)
    test_tensor_365 = slide_window(scaler_test.values, input_len = 90, output_len = 365, target_col_index = target_col_index_test)
    train_loader_365 = DataLoader(train_tensor_365, batch_size = 64, shuffle = True)
    test_loader_365 = DataLoader(test_tensor_365, batch_size = 64, shuffle = False)


    print("="*20, "LSTMmodel-365days", "="*20)
    lstm_mse_scores_365days, lstm_mae_scores_365days, lstm_mape_scores_365days, lstm_r2_scores_365days = multi_runs(model_name="lstm",runs=runs, epochs=epochs, lr=1e-6, output_len=365, input_dim=input_dim, train_loader=train_loader_365, test_loader=test_loader_365)
    # --- 结果汇总 ---
    print("\n--- Final LSTM Results ---")
    print(f"Prediction Length: {365} days")
    print(f"Average MSE: {np.mean(lstm_mse_scores_365days):.6f} ± {np.std(lstm_mse_scores_365days):.6f}")
    print(f"Average MAE: {np.mean(lstm_mae_scores_365days):.6f} ± {np.std(lstm_mae_scores_365days):.6f}")
    print(f"Average MAPE: {np.mean(lstm_mape_scores_365days):.6f} ± {np.std(lstm_mape_scores_365days):.6f}")
    print(f"Average R2: {np.mean(lstm_r2_scores_365days):.6f} ± {np.std(lstm_r2_scores_365days):.6f}")

    
    print("="*20, "Transformermodel-365days", "="*20)
    transformer_mse_scores_365days, transformer_mae_scores_365days, transformer_mape_scores_365days, transformer_r2_scores_365days = multi_runs(model_name="transformer",runs=runs, epochs=epochs, lr=1e-4, output_len=365, input_dim=input_dim, train_loader=train_loader_365, test_loader=test_loader_365)
    # --- 结果汇总 ---
    print("\n--- Final Transformer Results ---")
    print(f"Prediction Length: {365} days")
    print(f"Average MSE: {np.mean(transformer_mse_scores_365days):.6f} ± {np.std(transformer_mse_scores_365days):.6f}")
    print(f"Average MAE: {np.mean(transformer_mae_scores_365days):.6f} ± {np.std(transformer_mae_scores_365days):.6f}")
    print(f"Average MAPE: {np.mean(transformer_mape_scores_365days):.6f} ± {np.std(transformer_mape_scores_365days):.6f}")
    print(f"Average R2: {np.mean(transformer_r2_scores_365days):.6f} ± {np.std(transformer_r2_scores_365days):.6f}")

    
    print("="*20, "WG-TCN-Transformer-365days", "="*20)
    wg_mse_scores_365days, wg_mae_scores_365days, wg_mape_scores_365days, wg_r2_scores_365days = multi_runs(model_name="wg_tcn_transformer", runs=runs, epochs=epochs, lr=3e-6, output_len=365, input_dim=input_dim, train_loader=train_loader_365, test_loader=test_loader_365, weather_indices=weather_indices)
    
    # --- 结果汇总 ---
    print(f"\n--- Final WG-TCN-Transformer  Results ({365} days) ---")
    print(f"Average MSE: {np.mean(wg_mse_scores_365days):.6f} ± {np.std(wg_mse_scores_365days):.6f}")
    print(f"Average MAE: {np.mean(wg_mae_scores_365days):.6f} ± {np.std(wg_mae_scores_365days):.6f}")
    print(f"Average MAPE: {np.mean(wg_mape_scores_365days):.6f} ± {np.std(wg_mape_scores_365days):.6f}")
    print(f"Average R2: {np.mean(wg_r2_scores_365days):.6f} ± {np.std(wg_r2_scores_365days):.6f}")
