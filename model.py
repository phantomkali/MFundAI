import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim

# Read the CSV file
df = pd.read_csv('comprehensive_mutual_funds_data.csv')

# Convert numeric columns to float
numeric_columns = ['sortino', 'alpha', 'sd', 'beta', 'sharpe', 'returns_1yr', 'returns_3yr', 'returns_5yr']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values based on skewness
for col in numeric_columns:
    skewness = df[col].skew()
    if pd.notnull(skewness) and skewness > 1: # type: ignore
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mean())

# Encode categorical variables (assuming they exist in the dataset)
categorical_columns = ['category', 'sub_category', 'amc_name', 'fund_manager']
label_encoders = {}
for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        numeric_columns.append(col)  # Add encoded columns to features

# Feature Engineering
df['risk_return_score'] = (df['returns_1yr'] + df['returns_3yr'] + df['returns_5yr']) / df['risk_level']
numeric_columns.append('risk_return_score')

# Split into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Preprocessing function
def super_preprocess(features, targets):
    feature_scaler = RobustScaler()
    features_scaled = feature_scaler.fit_transform(features)
    
    target_scaler = StandardScaler()
    targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1))
    
    # Simplified feature engineering: quadratic terms and EMA
    features_poly = np.hstack([features_scaled, np.power(features_scaled, 2)])
    
    ema_features = np.copy(features_scaled)
    alpha = 0.3
    for i in range(1, len(features_scaled)):
        ema_features[i] = alpha * features_scaled[i] + (1 - alpha) * ema_features[i-1]
    
    features_poly = np.hstack([features_poly, ema_features])
    
    return torch.FloatTensor(features_poly), torch.FloatTensor(targets_scaled)

# Accuracy calculation with R²
def calculate_accuracy(model, data_loader):
    model.eval()
    total = 0
    mse = 0
    predictions = []
    actuals = []
    
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3]
    correct_at_threshold = {t: 0 for t in thresholds}
    
    with torch.no_grad():
        for features, targets in data_loader:
            outputs = model(features)
            mse += nn.MSELoss()(outputs, targets).item() * targets.size(0)
            total += targets.size(0)
            
            predictions.extend(outputs.numpy().flatten())
            actuals.extend(targets.numpy().flatten())
            
            for t in thresholds:
                correct_at_threshold[t] += torch.sum(torch.abs(outputs - targets) < t).item() # type: ignore
    
    mse = mse / total
    accuracies = {t: (correct_at_threshold[t] / total) * 100 for t in thresholds}
    r2 = r2_score(actuals, predictions)
    
    return accuracies[0.15], mse, accuracies, r2

# Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.fc = nn.Linear(dim, dim)
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = self.fc(out)
        return out + x  # Residual connection

# Simplified SuperModel
class SuperModel(nn.Module):
    def __init__(self, input_dim):
        super(SuperModel, self).__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.attention = AttentionBlock(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.3)
        self.skip1 = nn.Linear(input_dim, 128)
        self.skip2 = nn.Linear(128, 256)
        self.skip3 = nn.Linear(256, 128)
        
    def forward(self, x):
        x = self.input_bn(x)
        
        identity = self.skip1(x)
        x = F.silu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x += identity
        
        identity = self.skip2(x)
        x = F.silu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x += identity
        
        x = self.attention(x)
        
        identity = self.skip3(x)
        x = F.silu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x += identity
        
        x = self.fc4(x)
        return x

# Local training function
def super_train_on_client(model, data_loader, epochs=150):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) # type: ignore
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5) # type: ignore
    
    mse_criterion = nn.MSELoss()
    huber_criterion = nn.HuberLoss(delta=0.5)
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        total_loss = 0
        for x, y in data_loader:
            optimizer.zero_grad()
            output = model(x)
            mse_loss = mse_criterion(output, y)
            huber_loss = huber_criterion(output, y)
            loss = 0.4 * mse_loss + 0.6 * huber_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(data_loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict().copy()
        
        if epoch % 25 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    model.load_state_dict(best_state)
    return model.state_dict()

# FedProx aggregation
def proximal_aggregate(client_updates, global_model_state, client_performances, mu=0.1):
    weights = [1/(p + 1e-6) for p in client_performances]
    total_weight = sum(weights)
    normalized_weights = [w/total_weight for w in weights]
    
    aggregated_model = {key: torch.zeros_like(value, dtype=torch.float32) 
                       for key, value in client_updates[0].items()}
    
    for i, update in enumerate(client_updates):
        for key in aggregated_model.keys():
            proximal_term = mu * (update[key].float() - global_model_state[key].float())
            aggregated_model[key] += (update[key].float() - proximal_term) * normalized_weights[i]
    
    return aggregated_model

# Federated Learning Setup
NUM_CLIENTS = 2

# Stratified split (assuming 'category' exists; adjust if not)
if 'category' in train_df.columns:
    sss = StratifiedShuffleSplit(n_splits=NUM_CLIENTS, test_size=1/NUM_CLIENTS, random_state=42)
    client_dataframes = []
    for train_idx, _ in sss.split(train_df, train_df['category']):
        client_dataframes.append(train_df.iloc[train_idx])
else:
    client_dataframes = np.array_split(train_df, NUM_CLIENTS)

# Prepare client data
client_data = []
for client_df in client_dataframes:
    features = client_df[numeric_columns].values # type: ignore
    targets = client_df['returns_1yr'].values # type: ignore
    features_scaled, targets_scaled = super_preprocess(features, targets)
    batch_size = max(16, min(48, len(features) // 2))
    dataset = TensorDataset(features_scaled, targets_scaled)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    client_data.append(data_loader)

# Prepare validation data
val_features = val_df[numeric_columns].values
val_targets = val_df['returns_1yr'].values
val_features_scaled, val_targets_scaled = super_preprocess(val_features, val_targets)
val_dataset = TensorDataset(val_features_scaled, val_targets_scaled)
val_loader = DataLoader(val_dataset, batch_size=64)

# Initialize model
input_dim = next(iter(client_data[0]))[0].shape[1]
global_model = SuperModel(input_dim=input_dim)

# Training loop
num_rounds = 5
best_accuracy = 0
best_mse = float('inf')
best_model_state = None
patience = 5
no_improve_count = 0

for round in range(num_rounds):
    client_updates = []
    client_accuracies = []
    client_mses = []
    
    for data_loader in client_data:
        local_model = SuperModel(input_dim=input_dim)
        local_model.load_state_dict(global_model.state_dict())
        client_update = super_train_on_client(local_model, data_loader)
        client_updates.append(client_update)
        
        accuracy, mse, _, _ = calculate_accuracy(local_model, data_loader)
        client_accuracies.append(accuracy)
        client_mses.append(mse)
    
    global_update = proximal_aggregate(client_updates, global_model.state_dict(), client_mses)
    global_model.load_state_dict(global_update)
    
    val_accuracy, val_mse, val_all_accuracies, val_r2 = calculate_accuracy(global_model, val_loader)
    
    if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_mse < best_mse):
        best_accuracy = val_accuracy
        best_mse = val_mse
        best_model_state = global_model.state_dict().copy()
        no_improve_count = 0
    else:
        no_improve_count += 1
    
    if no_improve_count >= patience:
        print(f"Early stopping at round {round+1}")
        break
    
    print(f"\nRound {round + 1}:")
    print(f"Avg Client Accuracy: {sum(client_accuracies)/len(client_accuracies):.2f}%")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Accuracy at Thresholds: {val_all_accuracies}")
    print(f"Validation MSE: {val_mse:.6f}, R²: {val_r2:.4f}")

# Load and evaluate best model
global_model.load_state_dict(best_model_state) # type: ignore
final_accuracy, final_mse, final_all_accuracies, final_r2 = calculate_accuracy(global_model, val_loader)
print("\nFinal Best Model Evaluation:")
print(f"Final Accuracy: {final_accuracy:.2f}%")
print(f"Accuracy at Thresholds: {final_all_accuracies}")
print(f"Final MSE: {final_mse:.4f}, R²: {final_r2:.4f}")

# Save the best model
torch.save(best_model_state, 'mutual_fund_model.pth')
print("Best model saved successfully!")
