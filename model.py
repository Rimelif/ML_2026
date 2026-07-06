import torch
import torch.nn as nn
import math


class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_len, num_layers=2):
            """
            初始化LSTM模型。

            参数:
            input_dim (int): 输入特征的数量。
            hidden_dim (int): LSTM隐藏层的维度。
            output_len (int): 预测序列的长度（输出维度）。
            num_layers (int): LSTM的层数。
            """
            super(LSTMModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True, dropout = 0.1)
            self.fc = nn.Linear(hidden_dim, output_len)
        
        def forward(self, x):
            # 初始化隐藏状态和细胞状态
            # h0 shape: (num_layers, batch_size, hidden_dim)
            # c0 shape: (num_layers, batch_size, hidden_dim)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            out, _ = self.lstm(x, (h0, c0))

            # 获取最后一个时间步的输出
            pred = self.fc(out[:, -1, :])
            return pred


# --- Positional Encoding for Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 创建位置编码矩阵pe，形状为[max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # position是一个形状为[max_len, 1]的张量，表示位置索引
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 在第0维增加一个维度，使其形状为[1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x是输入张量，形状为[batch_size, seq_len, embed_dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- Transformer Model Definition ---
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, d_hid, num_layers, output_len, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_fc = nn.Linear(d_model, output_len)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, input_dim]
        """
        src = self.input_fc(src)
        src = self.pos_emb(src)
        output = self.encoder(src)
        
        # We take the output of the last time step to make a prediction
        output = self.output_fc(output[:, -1, :])
        return output
    

class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNResidualBlock, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.downsample(x)


class WeatherGatedTCNTransformerModel(nn.Module):
    def __init__(self, input_dim, weather_indices, tcn_channels, kernel_size, d_model,
                 nhead, d_hid, num_transformer_layers, output_len, dropout=0.2):
        super(WeatherGatedTCNTransformerModel, self).__init__()
        weather_indices = list(weather_indices)
        weather_set = set(weather_indices)
        power_indices = [i for i in range(input_dim) if i not in weather_set]

        self.register_buffer("weather_indices", torch.tensor(weather_indices, dtype=torch.long), persistent=False)
        self.register_buffer("power_indices", torch.tensor(power_indices, dtype=torch.long), persistent=False)

        tcn_layers = []
        in_channels = len(power_indices)
        for layer_id, out_channels in enumerate(tcn_channels):
            dilation = 2 ** layer_id
            tcn_layers.append(TCNResidualBlock(in_channels, out_channels, kernel_size, dilation, dropout))
            in_channels = out_channels
        self.tcn = nn.Sequential(*tcn_layers)

        self.input_projection = nn.Linear(tcn_channels[-1], d_model)
        self.weather_projection = nn.Linear(len(weather_indices), d_model)
        self.weather_gate = nn.Sequential(
            nn.Linear(len(weather_indices), d_model),
            nn.Sigmoid(),
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        self.output_fc = nn.Linear(d_model, output_len)

    def forward(self, src):
        # src: [batch, seq_len, input_dim]
        power_x = torch.index_select(src, dim=2, index=self.power_indices)
        weather_x = torch.index_select(src, dim=2, index=self.weather_indices)

        tcn_input = power_x.permute(0, 2, 1)
        tcn_output = self.tcn(tcn_input).permute(0, 2, 1)
        hidden = self.input_projection(tcn_output)

        weather_context = self.weather_projection(weather_x)
        gate = self.weather_gate(weather_x)
        hidden = hidden * (1.0 + gate) + weather_context

        transformer_input = self.pos_encoder(hidden)
        transformer_output = self.transformer_encoder(transformer_input)
        output = self.output_fc(transformer_output[:, -1, :])
        return output
