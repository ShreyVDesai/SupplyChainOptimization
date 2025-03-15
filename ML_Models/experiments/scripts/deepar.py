import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

#############################################
# CONFIGURATIONS
#############################################
CSV_FILE = 'C:/Users/shrey/Projects/SupplyChainOptimization/data/transactions_20230103_20241231.csv'  # <-- Change to your CSV file path
DATE_COL = "Date"
PRODUCT_COL = "Product Name"
TARGET_COL = "Total Quantity"

WINDOW_SIZE = 20
TRAIN_RATIO = 0.8
BATCH_SIZE = 16
EPOCHS = 10
EMBEDDING_DIM = 16
HIDDEN_SIZE = 32
LR = 0.001
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

#############################################
# 1) LOAD AND PREPARE THE DATA
#############################################
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    
    # Convert Date to datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="mixed")
    
    # Sort by date (and product if desired)
    df = df.sort_values(by=[DATE_COL, PRODUCT_COL]).reset_index(drop=True)
    return df

df = load_data(CSV_FILE)

# Encode products
product_encoder = LabelEncoder()
df[PRODUCT_COL] = product_encoder.fit_transform(df[PRODUCT_COL])

# Scale the target (Total Quantity)
scaler = MinMaxScaler()
df[TARGET_COL] = scaler.fit_transform(df[[TARGET_COL]])

#############################################
# 2) SPLIT INTO TRAIN AND TEST
#############################################
# We'll do a date-based split, taking the first 80% of dates for training.
unique_dates = df[DATE_COL].sort_values().unique()
split_index = int(len(unique_dates) * TRAIN_RATIO)
train_dates = unique_dates[:split_index]
test_dates = unique_dates[split_index:]

train_df = df[df[DATE_COL].isin(train_dates)].copy()
test_df  = df[df[DATE_COL].isin(test_dates)].copy()

#############################################
# 3) CREATE TRAINING DATASET
#    We create sliding windows across *all* products/time.
#############################################
class TimeSeriesDataset(Dataset):
    """
    Creates sliding windows across the entire (product, date) space.
    Each sample is:
      - product sequence over WINDOW_SIZE timesteps
      - quantity sequence over WINDOW_SIZE timesteps
      - next target (the next quantity after the window)
    """
    def __init__(self, df, window_size=20):
        super().__init__()
        self.window_size = window_size
        
        # We'll store (product_seq, quantity_seq, next_quantity)
        self.samples = []
        
        # Group by product so each product has its own time series
        grouped = df.groupby(PRODUCT_COL, group_keys=True)
        
        for product_id, group in grouped:
            # Sort by date to ensure correct time order
            group = group.sort_values(by=DATE_COL)
            quantities = group[TARGET_COL].values
            products = group[PRODUCT_COL].values
            
            # Create sliding windows
            if len(group) <= window_size:
                # Not enough data points for a full window
                continue
            
            for i in range(len(group) - window_size):
                past_products = products[i : i + window_size]
                past_quantities = quantities[i : i + window_size]
                next_quantity = quantities[i + window_size]
                
                self.samples.append((past_products, past_quantities, next_quantity))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        p_seq, q_seq, next_q = self.samples[idx]
        
        # Convert to tensors
        p_seq = torch.tensor(p_seq, dtype=torch.long)    # product IDs
        q_seq = torch.tensor(q_seq, dtype=torch.float32) # quantity values
        next_q = torch.tensor(next_q, dtype=torch.float32)
        
        return p_seq, q_seq, next_q

train_dataset = TimeSeriesDataset(train_df, WINDOW_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

#############################################
# 4) DEFINE THE MODEL (DeepAR-like)
#############################################
class DeepAR(nn.Module):
    def __init__(self, num_products, embedding_dim, hidden_size):
        super(DeepAR, self).__init__()
        
        # Embedding for product ID
        self.embedding = nn.Embedding(num_products, embedding_dim)
        
        # LSTM takes (embedding + 1) as input_size (the +1 is for quantity)
        input_size = embedding_dim + 1
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Final fully-connected layer
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, product_seq, quantity_seq):
        """
        product_seq: (batch, window_size)
        quantity_seq: (batch, window_size)
        """
        # Embed product IDs
        embedded = self.embedding(product_seq)  # (batch, window_size, embedding_dim)
        
        # Concatenate embedded product + quantity
        x = torch.cat([embedded, quantity_seq.unsqueeze(-1)], dim=-1)  
        # x shape: (batch, window_size, embedding_dim+1)
        
        # Pass through LSTM
        output, _ = self.lstm(x)  # (batch, window_size, hidden_size)
        
        # Take the last time step
        last_hidden = output[:, -1, :]  # (batch, hidden_size)
        
        # FC to predict next quantity
        prediction = self.fc(last_hidden)  # (batch, 1)
        
        return prediction

num_products = len(product_encoder.classes_)
model = DeepAR(num_products, EMBEDDING_DIM, HIDDEN_SIZE)

#############################################
# 5) TRAIN THE MODEL
#############################################
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for p_seq, q_seq, next_q in train_loader:
        optimizer.zero_grad()
        
        pred = model(p_seq, q_seq).squeeze()  # shape: (batch,)
        loss = criterion(pred, next_q)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

#############################################
# 6) EVALUATION: FORECAST PER PRODUCT, COMPUTE RMSE
#############################################

def forecast_product(model, product_id, product_df, window_size=20):
    """
    Iteratively forecast the test portion of a single product.
    product_df is the sorted data (by date) for that product only (test set).
    
    We'll assume we can take the *end* of the train set + any needed overlap
    from test for 'rolling' forecasts. For simplicity, we'll just do 
    iterative forecasting across the test set:
    
    1. Start with the last window_size points from the *train+test* data 
       that precede the test portion. 
    2. Predict the next point, compare with the actual test point.
    3. Append the actual test point to the sequence for the next iteration 
       (teacher-forcing style).
    """
    # Sort by date
    product_df = product_df.sort_values(by=DATE_COL)
    all_quantities = product_df[TARGET_COL].values
    all_products = product_df[PRODUCT_COL].values
    
    predictions = []
    actuals = []
    
    # If there's not enough data to form at least one window, skip
    if len(all_quantities) <= window_size:
        return None, None  # no predictions
    
    # We'll maintain a rolling window of the *true* values 
    # (to replicate a scenario where each new actual arrives).
    window_products = all_products[:window_size]
    window_quantities = all_quantities[:window_size]
    
    for i in range(window_size, len(all_quantities)):
        # Predict next step
        p_seq = torch.tensor(window_products, dtype=torch.long).unsqueeze(0)       # (1, window_size)
        q_seq = torch.tensor(window_quantities, dtype=torch.float32).unsqueeze(0) # (1, window_size)
        
        with torch.no_grad():
            pred = model(p_seq, q_seq).item()
        
        # Store prediction and actual
        predictions.append(pred)
        actuals.append(all_quantities[i])
        
        # Slide the window forward by 1 (append the *actual* value to mimic real-time updates)
        window_products = np.roll(window_products, -1)
        window_products[-1] = all_products[i]  # product ID stays the same for this product anyway
        
        window_quantities = np.roll(window_quantities, -1)
        window_quantities[-1] = all_quantities[i]  # we add the actual, not the predicted
    
    return predictions, actuals

model.eval()

per_product_rmses = []
unique_product_ids = test_df[PRODUCT_COL].unique()

for prod_id in unique_product_ids:
    # Combine train+test for the same product to get a full timeline
    # so we can start the window from the end of training.
    full_prod_df = df[df[PRODUCT_COL] == prod_id]
    # But we only compute RMSE on the test portion
    test_prod_df = test_df[test_df[PRODUCT_COL] == prod_id]
    
    if test_prod_df.empty:
        # No test data for this product
        continue
    
    # Forecast
    preds, actuals = forecast_product(model, prod_id, test_prod_df, WINDOW_SIZE)
    
    if preds is None or actuals is None:
        continue
    
    # Inverse transform the predictions & actuals
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    
    # Compute RMSE
    rmse = math.sqrt(mean_squared_error(actuals, preds))
    per_product_rmses.append(rmse)

if len(per_product_rmses) == 0:
    print("No products had sufficient test data to evaluate.")
else:
    final_rmse = np.mean(per_product_rmses)
    print(f"\nPer-product RMSEs: {per_product_rmses}")
    print(f"Final Average RMSE Across All Products: {final_rmse:.4f}")
