# %%
#Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# %%
# --- Load data ---
df = pd.read_csv('chn1(1).csv', encoding='utf-8-sig')
df = df.reset_index(drop=True)
step_seconds = 300
df['ElapsedSeconds'] = np.arange(len(df)) * step_seconds

# %%
# --- Only keep the last N points (focus on recent rise/drop) ---
N_recent = 30  # About last 2.5 hours # N can be adjusted accordingly
df_recent = df.iloc[-N_recent:].copy()
df_recent = df_recent.reset_index(drop=True)

# %%
# --- Feature engineering ---
max_lag = 10
for lag in range(1, max_lag+1):
    df_recent[f'lag_{lag}'] = df_recent['chn_1'].shift(lag)
for w in [3, 7, max_lag]:
    df_recent[f'roll{w}_mean'] = df_recent['chn_1'].rolling(w).mean()
    df_recent[f'roll{w}_std'] = df_recent['chn_1'].rolling(w).std()

def rolling_slope(x):
    idx = np.arange(len(x))
    if len(x) < 2:
        return 0
    coef = np.polyfit(idx, x, 1)
    return coef[0]

trend_window = 7
df_recent['roll_trend'] = df_recent['chn_1'].rolling(trend_window).apply(rolling_slope, raw=True)

df_recent = df_recent.dropna().reset_index(drop=True)

feature_cols = [c for c in df_recent.columns if c not in ['Time','chn_1']]
X = df_recent[feature_cols]
y = df_recent['chn_1']

# %%
# --- Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# --- Train-test split ---
split_idx = int(len(df_recent)*0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# %%
# --- Ridge regression ---
model = Ridge()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_all = model.predict(X_scaled)

# %%
# --- Print R2 and MSE ---
overall_r2 = r2_score(y, y_pred_all)
overall_mse = mean_squared_error(y, y_pred_all)
print(f"Overall R2: {overall_r2:.4f}")
print(f"Overall MSE: {overall_mse:.6f}")

# --- Print correlations ---
train_corr = np.corrcoef(y_train, y_pred_train)[0,1]
test_corr = np.corrcoef(y_test, y_pred_test)[0,1]
print(f"Train set correlation: {train_corr:.4f}")
print(f"Test set correlation:  {test_corr:.4f}")

# %%
# --- Visualize train/test correlation ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(y_train, y_pred_train, alpha=0.8, color='b')
plt.xlabel("True (train)")
plt.ylabel("Predicted (train)")
plt.title(f"Train correlation: {train_corr:.3f}")

plt.subplot(1,2,2)
plt.scatter(y_test, y_pred_test, alpha=0.8, color='g')
plt.xlabel("True (test)")
plt.ylabel("Predicted (test)")
plt.title(f"Test correlation: {test_corr:.3f}")
plt.tight_layout()
plt.show()

# %%
# --- Predict future 2 hrs (24 steps) ---
future_steps = 24
future_rows = []
last_known = df_recent.iloc[-1:].copy()
for i in range(future_steps):
    new_row = last_known.copy()
    new_row['ElapsedSeconds'] += step_seconds
    # Shift lags
    for lag in range(max_lag, 1, -1):
        new_row[f'lag_{lag}'] = new_row[f'lag_{lag-1}']
    new_row['lag_1'] = last_known['chn_1'].values[0]
    # Rolling features
    recent_vals = [new_row[f'lag_{k}'].values[0] for k in range(1, max_lag+1)]
    for w in [3, 7, max_lag]:
        vals = recent_vals[:w]
        new_row[f'roll{w}_mean'] = np.mean(vals)
        new_row[f'roll{w}_std'] = np.std(vals)
    trend_vals = [new_row[f'lag_{k}'].values[0] for k in range(1, trend_window+1)]
    new_row['roll_trend'] = rolling_slope(np.array(trend_vals[::-1]))
    X_future = scaler.transform(new_row[feature_cols])
    pred = model.predict(X_future)[0]
    new_row['chn_1'] = pred
    future_rows.append(new_row)
    last_known = new_row

future_df = pd.concat(future_rows, ignore_index=True)

# %%
# --- Print future voltages with Time (minutes) and Voltage ---
out_df = pd.DataFrame({
    'Time (min)': future_df['ElapsedSeconds']/60,
    'Predicted_Voltage': future_df['chn_1']
})
print("\nFuture voltage prediction:")
print(out_df.to_string(index=False))

# %%
# --- Plot ---
plt.figure(figsize=(15,7))
plt.plot(df['ElapsedSeconds']/60, df['chn_1'], 'b.-', label='All Historical')
plt.plot(df_recent['ElapsedSeconds']/60, df_recent['chn_1'], 'go-', label='Recent (Training & Test)')
plt.plot(future_df['ElapsedSeconds']/60, future_df['chn_1'], 'rx--', label='Predicted Future (2hr)')
plt.xlabel('Elapsed Time (minutes)')
plt.ylabel('Voltage (chn_1)')
plt.title(f'Soil MFC Voltage: Ridge with Trend Feature')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Select only numeric columns for correlation
corr_features = df_recent[[c for c in df_recent.columns if c not in ['Time']]]

plt.figure(figsize=(12, 10))
sns.heatmap(corr_features.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


