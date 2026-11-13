import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --------------------------
# 1. Core Configuration & Parameters
# --------------------------
torch.manual_seed(50)
np.random.seed(50)

# Dataset path
DATA_PATH = r"C:\Users\Administrator\Your dataset.csv"

# Core hyperparameters
WINDOW_SIZE = 
EPOCHS = 
LEARNING_RATE = 
EARLY_STOPPING_PATIENCE = 
TRAIN_RATIO = 

# Physical parameters
L =     # Bonded length
sigma =    # Loading stress
h =       # Bonded thickness
SLOWDOWN_BASE = 

# Initial optimal parameters
BEST_A = 
BEST_n = 
BEST_m =
BEST_c = 
BEST_k = 
BEST_alpha = 
BEST_beta = 

# Parameter bounds constraint
PARAM_BOUNDS = {
    'A': (),
    'n': (),
    'm': (),
    'c': (),
    'k': (),
    'alpha': (),
    'beta': ()
}

# --------------------------
# 2. Data Loading & Preprocessing (Core)
# --------------------------
def load_and_preprocess_data(data_path, window_size, train_ratio):
    # Load raw data
    df = pd.read_csv(data_path)
    raw_time = df['time'].values.reshape(-1, 1)
    raw_creep = df['creep'].values.reshape(-1, 1)
    
    # Normalization
    scaler_creep = MinMaxScaler(feature_range=(0, 1))
    creep_scaled = scaler_creep.fit_transform(raw_creep)
    
    scaler_time = MinMaxScaler(feature_range=(0, 1))
    time_scaled = scaler_time.fit_transform(raw_time).flatten()
    
    # Create time series sequences
    def create_sequences(data, time_arr, win_size):
        X, y, t = [], [], []
        for i in range(len(data) - win_size):
            X.append(data[i:i+win_size])
            y.append(data[i+win_size])
            t.append(time_arr[i+win_size])
        return np.array(X), np.array(y).reshape(-1, 1), np.array(t).reshape(-1, 1)
    
    X, y, t = create_sequences(creep_scaled, time_scaled, window_size)
    
    # Split train and test set
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    t_train, t_test = t[:train_size], t[train_size:]
    
    return (X_train, y_train, t_train, X_test, y_test, t_test, 
            scaler_creep, scaler_time, raw_time, raw_creep)

# Load dataset
X_train, y_train, t_train, X_test, y_test, t_test, scaler_creep, scaler_time, raw_time, raw_creep = load_and_preprocess_data(
    DATA_PATH, WINDOW_SIZE, TRAIN_RATIO
)

# --------------------------
# 3. Tensor Conversion & Device Configuration
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert to torch tensor
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
t_train_tensor = torch.FloatTensor(t_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)
t_test_tensor = torch.FloatTensor(t_test).to(device)

# --------------------------
# 4. Core Model Definition (LSTM + PINN)
# --------------------------
class EnhancedLSTM(nn.Module):
    """Basic LSTM prediction module"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class PINN(nn.Module):
    """PINN with LSTM, power-law constraint and parameter bounds"""
    def __init__(self, lstm_model, L, sigma, h, slowdown_base, param_bounds):
        super().__init__()
        self.lstm = lstm_model
        self.param_bounds = param_bounds
        
        # Fixed physical parameters
        self.L = torch.tensor([[L]], dtype=torch.float32, device=device)
        self.sigma = torch.tensor([[sigma]], dtype=torch.float32, device=device)
        self.h = torch.tensor([[h]], dtype=torch.float32, device=device)
        self.slowdown_base = slowdown_base
        
        # Learnable parameters initialized with optimal values
        self.A = nn.Parameter(torch.tensor([[BEST_A]], dtype=torch.float32, device=device))
        self.n = nn.Parameter(torch.tensor([[BEST_n]], dtype=torch.float32, device=device))
        self.m = nn.Parameter(torch.tensor([[BEST_m]], dtype=torch.float32, device=device))
        self.c = nn.Parameter(torch.tensor([[BEST_c]], dtype=torch.float32, device=device))
        self.k = nn.Parameter(torch.tensor([[BEST_k]], dtype=torch.float32, device=device))
        self.alpha = nn.Parameter(torch.tensor([[BEST_alpha]], dtype=torch.float32, device=device))
        self.beta = nn.Parameter(torch.tensor([[BEST_beta]], dtype=torch.float32, device=device))

    def clamp_parameters(self):
        """Constrain parameters within predefined bounds"""
        self.A.data = torch.clamp(self.A.data, *self.param_bounds['A'])
        self.n.data = torch.clamp(self.n.data, *self.param_bounds['n'])
        self.m.data = torch.clamp(self.m.data, *self.param_bounds['m'])
        self.c.data = torch.clamp(self.c.data, *self.param_bounds['c'])
        self.k.data = torch.clamp(self.k.data, *self.param_bounds['k'])
        self.alpha.data = torch.clamp(self.alpha.data, *self.param_bounds['alpha'])
        self.beta.data = torch.clamp(self.beta.data, *self.param_bounds['beta'])

    def forward(self, x, t, is_training=True):
        """Forward propagation: LSTM prediction + physics constraint"""
        t = t.requires_grad_(is_training)
        
        # Parameter constraint in training phase
        if is_training:
            self.clamp_parameters()
        
        # LSTM prediction + time correction terms
        creep_pred_lstm = self.lstm(x)
        creep_pred = creep_pred_lstm + self.alpha * t + self.beta * torch.sqrt(t + 1e-8)
        
        # Calculate theoretical creep rate with slowdown factor
        L_tensor = self.L.expand_as(creep_pred)
        sigma_tensor = self.sigma.expand_as(creep_pred)
        h_tensor = self.h.expand_as(creep_pred)
        slowdown_factor = torch.exp(-t / (self.slowdown_base * self.k))
        
        epsilon_dot_theory = self.A * (sigma_tensor ** self.n) * (h_tensor ** self.m) * \
                            (1 + self.c / L_tensor) * (t ** self.k) * slowdown_factor
        
        # Calculate physics loss (only in training)
        physics_loss = torch.tensor(0.0, device=device)
        if is_training:
            epsilon_dot_pred = torch.autograd.grad(
                outputs=creep_pred,
                inputs=t,
                grad_outputs=torch.ones_like(creep_pred, device=device),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            # Avoid None gradient
            epsilon_dot_pred = epsilon_dot_pred if epsilon_dot_pred is not None else torch.zeros_like(creep_pred)
            physics_loss = torch.mean((epsilon_dot_pred - epsilon_dot_theory) ** 2) * 1e5
        
        return creep_pred, physics_loss

    def simulate_creep(self, time_data):
        """Simulate creep curve with optimized parameters"""
        self.eval()
        with torch.no_grad():
            # Time normalization
            time_scaled = scaler_time.transform(time_data)
            t_tensor = torch.FloatTensor(time_scaled).reshape(-1, 1).to(device)
            
            # Expand physical parameters to batch size
            L_tensor = self.L.expand(len(t_tensor), 1)
            sigma_tensor = self.sigma.expand(len(t_tensor), 1)
            h_tensor = self.h.expand(len(t_tensor), 1)
            
            # Calculate creep rate with slowdown factor
            slowdown_factor = torch.exp(-t_tensor / (self.slowdown_base * self.k))
            epsilon_dot = self.A * (sigma_tensor ** self.n) * (h_tensor ** self.m) * \
                          (1 + self.c / L_tensor) * (t_tensor ** self.k) * slowdown_factor
            
            # Trapezoidal integration to get creep displacement
            epsilon_base = torch.zeros_like(epsilon_dot)
            for i in range(1, len(epsilon_dot)):
                dt = t_tensor[i] - t_tensor[i-1]
                epsilon_base[i] = epsilon_base[i-1] + (epsilon_dot[i-1] + epsilon_dot[i]) * dt / 2
            
            # Add time correction terms and inverse normalization
            creep_sim = epsilon_base + self.alpha * t_tensor + self.beta * torch.sqrt(t_tensor + 1e-8)
            creep_sim_actual = scaler_creep.inverse_transform(creep_sim.cpu().numpy())
            return creep_sim_actual

# --------------------------
# 5. Evaluation Metrics Calculation
# --------------------------
def calculate_evaluation_metrics(y_actual, y_pred):
    """Calculate 6 core regression metrics: MSE, RMSE, MAE, MAPE, R², RMSLE"""
    y_actual_flat = y_actual.flatten()
    y_pred_flat = y_pred.flatten()
    eps = 1e-8  # Avoid division by zero and log of zero
    
    mse = mean_squared_error(y_actual_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual_flat, y_pred_flat)
    mape = np.mean(np.abs((y_actual_flat - y_pred_flat) / (y_actual_flat + eps))) * 100
    r2 = r2_score(y_actual_flat, y_pred_flat)
    rmsle = np.sqrt(mean_squared_error(np.log(y_actual_flat + eps), np.log(y_pred_flat + eps)))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'RMSLE': rmsle
    }

# --------------------------
# 6. Model Initialization & Training (Core)
# --------------------------
# Initialize model
lstm = EnhancedLSTM(hidden_size=64, num_layers=1).to(device)
pinn_model = PINN(lstm, L=L, sigma=sigma, h=h, slowdown_base=SLOWDOWN_BASE, param_bounds=PARAM_BOUNDS).to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(pinn_model.parameters(), lr=LEARNING_RATE)
data_criterion = nn.MSELoss()

# Training hyperparameters for physics loss weight
beta_start = 
beta_end = 
best_total_loss = np.inf
no_improvement_count = 0

# Core training loop
for epoch in range(EPOCHS):
    pinn_model.train()
    optimizer.zero_grad()
    
    # Dynamic adjust physics loss weight
    beta = beta_start + (beta_end - beta_start) * (epoch / EPOCHS)
    
    # Forward pass
    creep_pred_train, physics_loss = pinn_model(X_train_tensor, t_train_tensor, is_training=True)
    
    # Calculate loss and backpropagation
    data_loss = data_criterion(creep_pred_train, y_train_tensor)
    total_loss = data_loss + beta * physics_loss
    total_loss.backward()
    optimizer.step()
    
    # Re-clamp parameters to avoid out of bounds after optimization
    pinn_model.clamp_parameters()
    
    # Early stopping mechanism
    if total_loss.item() < best_total_loss:
        best_total_loss = total_loss.item()
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        if no_improvement_count >= EARLY_STOPPING_PATIENCE:
            break

# --------------------------
# 7. Model Evaluation (Core)
# --------------------------
pinn_model.eval()

# Evaluate on training set
with torch.no_grad():
    train_pred, _ = pinn_model(X_train_tensor, t_train_tensor, is_training=False)
y_train_pred = scaler_creep.inverse_transform(train_pred.cpu().numpy())
y_train_actual = scaler_creep.inverse_transform(y_train)
train_metrics = calculate_evaluation_metrics(y_train_actual, y_train_pred)

# Evaluate on test set
with torch.no_grad():
    test_pred, _ = pinn_model(X_test_tensor, t_test_tensor, is_training=False)
y_test_pred = scaler_creep.inverse_transform(test_pred.cpu().numpy())
y_test_actual = scaler_creep.inverse_transform(y_test)
test_metrics = calculate_evaluation_metrics(y_test_actual, y_test_pred)

# Simulate creep curve and evaluate
creep_sim = pinn_model.simulate_creep(raw_time)
sim_r2 = r2_score(raw_creep.flatten(), creep_sim.flatten())