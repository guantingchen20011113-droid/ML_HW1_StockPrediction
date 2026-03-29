import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# 1. 數據獲取 (Data Acquisition)
# Download S&P 500 historical data
print("下載 S&P 500 數據中...")
ticker = "^GSPC"
start_date = "2021-01-01"
end_date = "2026-01-01" # Fetch up to 2025-12-31 inclusive
data = yf.download(ticker, start=start_date, end=end_date)

# 如果 yfinance 返回 MultiIndex 列，將其簡化
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# 2. 數據預處理 (Data Preprocessing) & 特徵工程 (Feature Engineering)
# 提取 'Close' 價格 (確保為單一列形式)
df = pd.DataFrame(data['Close']).copy()
if df.shape[1] > 1:
    df = df.iloc[:, [0]]
df.columns = ['Close']

# 加入特徵工程
# Lag_1: 前一天的收盤價
df['Lag_1'] = df['Close'].shift(1)

# MA_5 & MA_20: 5日與20日移動平均線
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()

# RSI: 相對強弱指標 (14期) - 手動計算以確保兼容性
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# 移除因計算特徵而產生的 NaN 值
df = df.dropna()


# 3. 數據切分 (Data Splitting)
# 依照時間順序切分：2021-2024 為訓練集，2025 為測試集
train_df = df[df.index.year < 2025]
test_df = df[df.index.year == 2025]

X_train = train_df[['Lag_1', 'MA_5', 'MA_20', 'RSI']]
y_train = train_df['Close']

X_test = test_df[['Lag_1', 'MA_5', 'MA_20', 'RSI']]
y_test = test_df['Close']

print(f"訓練集樣本數: {len(X_train)} (日期範疇: {train_df.index.min().date()} 至 {train_df.index.max().date()})")
print(f"測試集樣本數: {len(X_test)} (日期範疇: {test_df.index.min().date()} 至 {test_df.index.max().date()})")

# 4. 模型訓練與比較 (Model Training & Comparison)
# XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# 預測 (Prediction)
xgb_preds = xgb_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# 5. 評估 (Evaluation)
xgb_mse = mean_squared_error(y_test, xgb_preds)
rf_mse = mean_squared_error(y_test, rf_preds)

print(f"\n--- 模型評估結果 (MSE) ---")
print(f"XGBoost MSE: {xgb_mse:.4f}")
print(f"Random Forest MSE: {rf_mse:.4f}")

# 6. 視覺化 (Visualization)
# 設定美觀風格
plt.style.use('fivethirtyeight')

# 第一張圖: 股價預測比較圖
plt.figure(figsize=(14, 7))
plt.plot(test_df.index, y_test, label='Actual Price', color='black', linewidth=1.5)
plt.plot(test_df.index, xgb_preds, label='XGBoost Prediction', color='blue', linestyle='--', linewidth=1.2)
plt.plot(test_df.index, rf_preds, label='Random Forest Prediction', color='orange', linestyle='-.', linewidth=1.2)

plt.title('S&P 500 Index Prediction Comparison (2025)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('prediction_comparison.png')
print("已存儲: prediction_comparison.png")

# 第二張圖: MSE 比較圖
plt.figure(figsize=(10, 6))
models = ['XGBoost', 'Random Forest']
mse_values = [xgb_mse, rf_mse]
bars = plt.bar(models, mse_values, color=['royalblue', 'darkorange'], alpha=0.8)

plt.title('Model Comparison: Mean Squared Error (MSE)', fontsize=16)
plt.ylabel('MSE Value', fontsize=12)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('mse_comparison.png')
print("已存儲: mse_comparison.png")

plt.close('all')
