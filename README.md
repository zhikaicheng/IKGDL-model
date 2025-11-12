import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- Part 1: 数据加载模块 -----------------

def load_pretrain_data(file_path):
    """

    """
    print(f"Loading pre-training data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"预训练文件未找到: {file_path}")
        
    df = pd.read_excel(file_path)
    

    y_pre = df.iloc[:, 3].values.reshape(-1, 1)

    # 从第5列开始是气象特征数据 (索引为4)
    features_start_col = 4
    num_features = MET_TIMESTEPS * MET_FEATURES
    features_flat = df.iloc[:, features_start_col : features_start_col + num_features].values
    

    if features_flat.shape[1] != num_features:
        raise ValueError(f"预训练数据特征列数错误！预期 {num_features}, 实际 {features_flat.shape[1]}")


    x_pre = features_flat.reshape(-1, MET_TIMESTEPS, MET_FEATURES)


    x_mean = x_pre.mean(axis=(0, 1), keepdims=True)
    x_std = x_pre.std(axis=(0, 1), keepdims=True) + 1e-8
    x_pre_norm = (x_pre - x_mean) / x_std

    y_mean = y_pre.mean()
    y_std = y_pre.std() + 1e-8
    y_pre_norm = (y_pre - y_mean) / y_std
    
    print("Pre-training data loaded successfully.")
    return x_pre_norm, y_pre_norm, x_mean, x_std, y_mean, y_std

def load_finetune_or_validate_data(file_path, x_met_mean, x_met_std, y_mean, y_std):


    """
    print(f"Loading fine-tune/validate data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"微调/验证文件未找到: {file_path}")
        
    df = pd.read_excel(file_path)


    y_data = df.iloc[:, 3].values.reshape(-1, 1)


    met_start_col = 3
    num_met_features = MET_TIMESTEPS * MET_FEATURES
    met_flat = df.iloc[:, met_start_col : met_start_col + num_met_features].values
    if met_flat.shape[1] != num_met_features:
        raise ValueError(f"微调/验证数据的气象特征列数错误！预期 {num_met_features}, 实际 {met_flat.shape[1]}")
    x_met = met_flat.reshape(-1, MET_TIMESTEPS, MET_FEATURES)

    # --- 提取遥感数据 (rs) ---
    rs_start_col = met_start_col + num_met_features
    num_rs_features = RS_TIMESTEPS * RS_FEATURES
    rs_flat = df.iloc[:, rs_start_col : rs_start_col + num_rs_features].values
    if rs_flat.shape[1] != num_rs_features:
        raise ValueError(f"微调/验证数据的遥感特征列数错误！预期 {num_rs_features}, 实际 {rs_flat.shape[1]}")
    x_rs = rs_flat.reshape(-1, RS_TIMESTEPS, RS_FEATURES)

    # ----------- Debug: 打印shape和类型，防止float类型错误 -----------
    print("x_rs shape:", np.shape(x_rs), "type:", type(x_rs))
    if not (isinstance(x_rs, np.ndarray) and x_rs.ndim == 3):
        raise ValueError(f"x_rs 不是三维数组，实际类型: {type(x_rs)}, shape: {getattr(x_rs, 'shape', None)}, 值: {x_rs}")

    # --- 提取极端气候数据 (ec) ---
    ec_start_col = rs_start_col + num_rs_features
    num_ec_features = EC_FEATURES
    x_ec = df.iloc[:, ec_start_col : ec_start_col + num_ec_features].values
    if x_ec.shape[1] != num_ec_features:
        raise ValueError(f"微调/验证数据的极端气候特征列数错误！预期 {num_ec_features}, 实际 {x_ec.shape[1]}")

    # ----------- Debug: 打印shape和类型，防止float类型错误 -----------
    print("x_ec shape:", np.shape(x_ec), "type:", type(x_ec))
    if not (isinstance(x_ec, np.ndarray) and x_ec.ndim == 2):
        raise ValueError(f"x_ec 不是二维数组，实际类型: {type(x_ec)}, shape: {getattr(x_ec, 'shape', None)}, 值: {x_ec}")
    # ----------------------------------------------------------


    try:
        x_met = x_met.astype(float)
    except Exception as e:
        print("x_met 转换为 float 失败，异常值如下：")
        print(x_met)
        raise e

    try:
        x_rs = x_rs.astype(float)
    except Exception as e:
        print("x_rs 转换为 float 失败，异常值如下：")
        print(x_rs)
        raise e

    try:
        x_ec = x_ec.astype(float)
    except Exception as e:
        print("x_ec 转换为 float 失败，异常值如下：")
        print(x_ec)
        raise e

    try:
        y_data = y_data.astype(float)
    except Exception as e:
        print("y_data 转换为 float 失败，异常值如下：")
        print(y_data)
        raise e


    if np.isnan(x_rs).any():
        print("警告：x_rs 中存在 NaN，将用0填充！")
        x_rs = np.nan_to_num(x_rs, nan=0.0)
    if np.isnan(x_ec).any():
        print("警告：x_ec 中存在 NaN，将用0填充！")
        x_ec = np.nan_to_num(x_ec, nan=0.0)


    x_rs_mean = np.nanmean(x_rs, axis=(0, 1), keepdims=True)
    x_rs_std = np.nanstd(x_rs, axis=(0, 1), keepdims=True)
    x_rs_std = np.where((x_rs_std == 0) | np.isnan(x_rs_std), 1e-8, x_rs_std)
    x_rs_norm = (x_rs - x_rs_mean) / x_rs_std
    
    x_ec_mean = np.nanmean(x_ec, axis=0, keepdims=True)
    x_ec_std = np.nanstd(x_ec, axis=0, keepdims=True)
    x_ec_std = np.where((x_ec_std == 0) | np.isnan(x_ec_std), 1e-8, x_ec_std)
    x_ec_norm = (x_ec - x_ec_mean) / x_ec_std

    print("Fine-tune/validate data loaded successfully.")
    
    # 遥感数据需要填充以匹配气象数据的时间步长
    # (样本数, 10, 4) -> (样本数, 29, 4)
    x_rs_padded = np.zeros((x_rs_norm.shape[0], MET_TIMESTEPS, RS_FEATURES))
    x_rs_padded[:, -RS_TIMESTEPS:, :] = x_rs_norm # 将数据放在最后10个时间步
    
    x_met_norm = (x_met - x_met_mean) / x_met_std
    y_data_norm = (y_data - y_mean) / y_std
    
    return x_met_norm, x_rs_padded, x_ec_norm, y_data_norm, x_rs_mean, x_rs_std, x_ec_mean, x_ec_std


# ----------------- Part 2: KGDL 模型定义 -----------------
class KGDL(nn.Module):
    def __init__(self, met_dim, rs_dim, ec_dim, hidden_dim, lstm_layers):
        super(KGDL, self).__init__()
        self.met_dim = met_dim
        self.rs_dim = rs_dim
        self.ec_dim = ec_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(met_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

        self.iib_emb_linear = nn.Linear(met_dim, hidden_dim)
        self.iib_cov_linear = nn.Linear(rs_dim, hidden_dim)
        self.iib_ffn = nn.Sequential(
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.ReLU(),
            nn.Linear(hidden_dim, met_dim)
        )

        self.oib_out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.oib_cov_linear = nn.Linear(rs_dim, hidden_dim)
        self.oib_ec_linear = nn.Linear(ec_dim, hidden_dim)
        self.oib_ffn = nn.Sequential(
        nn.Linear(hidden_dim * 3, hidden_dim),
        nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_met, x_rs=None, x_ec=None, is_finetune=False):
        

        if not is_finetune:
            h_sequence, _ = self.lstm(x_met)
            h_last = h_sequence[:, -1, :]
            output = self.linear(h_last)
            return output


        else:
            if x_rs is None or x_ec is None:
                raise ValueError("微调阶段必须提供遥感(x_rs)和极端气候(x_ec)数据")

            # 1. IIB: 调整输入
            met_emb = self.iib_emb_linear(x_met)
            rs_emb_iib = self.iib_cov_linear(x_rs)
            combined_iib = torch.cat([met_emb, rs_emb_iib], dim=-1)
            adjustment = self.iib_ffn(combined_iib)
            x_met_adjusted = x_met + adjustment # 残差连接

            # 2. 核心LSTM
            h_sequence, _ = self.lstm(x_met_adjusted)
            h_last = h_sequence[:, -1, :]
            base_output = self.linear(h_last)


            h_emb_oib = self.oib_out_linear(h_last)
            rs_emb_oib = self.oib_cov_linear(x_rs).mean(dim=1) # 对时间步求平均
            ec_emb_oib = self.oib_ec_linear(x_ec)
            
            combined_oib = torch.cat([h_emb_oib, rs_emb_oib, ec_emb_oib], dim=-1)
            correction = self.oib_ffn(combined_oib)
            
            final_output = base_output + correction # 残差连接
            return final_output

# ----------------- Part 3: 训练和评估模块 -----------------

def pretrain_model(model, train_loader, num_epochs, lr):
    print("\n--- Starting Pre-training ---")
    criterion = nn.MSELoss()
    # 预训练只优化 LSTM 和 主输出层
    optimizer = torch.optim.Adam(
        list(model.lstm.parameters()) + list(model.linear.parameters()),
        lr=lr
    )
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_met=x_batch, is_finetune=False)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Pre-train Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        
    torch.save(model.state_dict(), PRETRAINED_MODEL_PATH)
    print(f"Pre-trained model saved to {PRETRAINED_MODEL_PATH}")

def finetune_model(model, train_loader, num_epochs, lr):
    print("\n--- Starting Fine-tuning ---")
    criterion = nn.MSELoss()
    # 微调时优化 IIB 和 OIB 模块
    optimizer = torch.optim.Adam(
        list(model.iib_emb_linear.parameters()) + list(model.iib_cov_linear.parameters()) +
        list(model.iib_ffn.parameters()) + list(model.oib_out_linear.parameters()) +
        list(model.oib_cov_linear.parameters()) + list(model.oib_ec_linear.parameters()) +
        list(model.oib_ffn.parameters()),
        lr=lr
    )


    for param in model.lstm.parameters():
        param.requires_grad = False
    for param in model.linear.parameters():
        param.requires_grad = False
        
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x_met_b, x_rs_b, x_ec_b, y_b in train_loader:
            x_met_b, x_rs_b, x_ec_b, y_b = x_met_b.to(device), x_rs_b.to(device), x_ec_b.to(device), y_b.to(device)

            optimizer.zero_grad()
            outputs = model(x_met=x_met_b, x_rs=x_rs_b, x_ec=x_ec_b, is_finetune=True)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Fine-tune Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")


    for param in model.parameters():
        param.requires_grad = True
        
    torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
    print(f"Fine-tuned model saved to {FINETUNED_MODEL_PATH}")

def evaluate_model(model, loader, y_std, y_mean, model_phase):
    print(f"\n--- Evaluating {model_phase} Model ---")
    model.eval()
    all_preds = []
    all_reals = []

    with torch.no_grad():
        for batch in loader:
            y_b = batch[-1]
            

            batch_device = [item.to(device) for item in batch]

            if model_phase == 'Pre-trained':

                outputs = model(x_met=batch_device[0], is_finetune=False)
            else: # Fine-tuned or Validating
                outputs = model(x_met=batch_device[0], x_rs=batch_device[1], x_ec=batch_device[2], is_finetune=True)


            preds_unnorm = outputs.cpu().numpy() * y_std + y_mean
            reals_unnorm = y_b.cpu().numpy() * y_std + y_mean
            
            all_preds.append(preds_unnorm)
            all_reals.append(reals_unnorm)
            
    all_preds = np.concatenate(all_preds)
    all_reals = np.concatenate(all_reals)
    
    r2 = r2_score(all_reals, all_preds)

    denominator = np.mean(np.array(all_reals))

    if denominator == 0:
        nrmse = np.inf
    else:
        nrmse = np.sqrt(mean_squared_error(all_reals, all_preds)) / denominator
    
    print(f"R-squared (R2): {r2:.4f}")
    print(f"Normalized Root Mean Squared Error (NRMSE): {nrmse:.4f}")
    
    return r2, nrmse

# ----------------- Part 4: 主执行流程 -----------------
def main():

    x_pre_norm, y_pre_norm, x_met_mean, x_met_std, y_mean, y_std = load_pretrain_data(PRETRAIN_FILE)
    

    pretrain_dataset = TensorDataset(torch.from_numpy(x_pre_norm).float(), torch.from_numpy(y_pre_norm).float())
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True)
    
    # 初始化模型
    model = KGDL(
        met_dim=MET_FEATURES,
        rs_dim=RS_FEATURES,
        ec_dim=EC_FEATURES,
        hidden_dim=128,
        lstm_layers=2
    ).to(device)


    pretrain_model(model, pretrain_loader, num_epochs=50, lr=0.001)
    

    # evaluate_model(model, pretrain_loader, y_std, y_mean, 'Pre-trained')

    # ---- 2. 微调 ----
    x_ft_met, x_ft_rs, x_ft_ec, y_ft, _, _, _, _ = load_finetune_or_validate_data(
        FINETUNE_FILE, x_met_mean, x_met_std, y_mean, y_std
    )
    
    # 创建数据加载器
    finetune_dataset = TensorDataset(
        torch.from_numpy(x_ft_met).float(),
        torch.from_numpy(x_ft_rs).float(),
        torch.from_numpy(x_ft_ec).float(),
        torch.from_numpy(y_ft).float()
    )
    finetune_loader = DataLoader(finetune_dataset, batch_size=32, shuffle=True)

    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    print(f"\nLoaded pre-trained weights from {PRETRAINED_MODEL_PATH} for fine-tuning.")
    

    finetune_model(model, finetune_loader, num_epochs=30, lr=0.0005)

    # ---- 3. 验证 ----
    print(f"\n--- Loading Validation Data from {VALIDATE_FILE} ---")
    x_val_met, x_val_rs, x_val_ec, y_val, _, _, _, _ = load_finetune_or_validate_data(
        VALIDATE_FILE, x_met_mean, x_met_std, y_mean, y_std
    )

    validate_dataset = TensorDataset(
        torch.from_numpy(x_val_met).float(),
        torch.from_numpy(x_val_rs).float(),
        torch.from_numpy(x_val_ec).float(),
        torch.from_numpy(y_val).float()
    )
    validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=False)
    
    model.load_state_dict(torch.load(FINETUNED_MODEL_PATH))
    print(f"\nLoaded fine-tuned weights from {FINETUNED_MODEL_PATH} for validation.")
    
    evaluate_model(model, validate_loader, y_std, y_mean, 'Validating')

if __name__ == '__main__':
    main()
