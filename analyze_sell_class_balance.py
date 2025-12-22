# -*- coding: utf-8 -*-
"""
分析 Sell Agent 訓練樣本的類別平衡
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ptrl_hybrid_system as hybrid

# 設定
SPLIT_DATE = '2023-01-01'
PROFIT_THRESHOLD = 1.10  # 10% 獲利門檻

print("=" * 60)
print("分析 Sell Agent 訓練樣本的類別平衡")
print("=" * 60)

# 載入 LSTM 模型
print("\n[1] 載入 LSTM 模型...")
hybrid.load_best_lstm_models()

# 載入資料
print("\n[2] 載入 ^TWII 資料...")
twii_raw = hybrid._load_local_twii_data(start_date="2000-01-01", end_date=SPLIT_DATE)
twii_df = hybrid.calculate_features(twii_raw, twii_raw, ticker="^TWII", use_cache=False)

print(f"  資料筆數: {len(twii_df)}")

# 模擬 SellEnvHybrid 的 episode 建構邏輯
print("\n[3] 分析 Episode 類別分布...")

buy_indices = np.where(twii_df['Signal_Buy_Filter'])[0]
close_prices = twii_df['Close'].values.astype(np.float32)

episodes_info = []
positive_count = 0
negative_count = 0

for idx in buy_indices:
    if idx + 120 < len(twii_df):
        episode_prices = close_prices[idx:idx+120]
        returns = episode_prices / episode_prices[0]
        max_return = np.max(returns)
        
        # 檢查這個 episode 是否能達到獲利門檻
        is_profitable = max_return >= PROFIT_THRESHOLD
        
        if is_profitable:
            positive_count += 1
        else:
            negative_count += 1
        
        episodes_info.append({
            'start_date': twii_df.index[idx],
            'max_return': max_return,
            'max_return_pct': (max_return - 1) * 100,
            'is_profitable': is_profitable
        })

total_episodes = positive_count + negative_count

print("\n" + "=" * 60)
print("結果")
print("=" * 60)
print(f"獲利門檻: {(PROFIT_THRESHOLD - 1) * 100:.0f}%")
print(f"總 Episode 數: {total_episodes}")
print(f"正樣本 (max_return >= {PROFIT_THRESHOLD:.0%}): {positive_count} ({positive_count/total_episodes*100:.1f}%)")
print(f"負樣本 (max_return < {PROFIT_THRESHOLD:.0%}): {negative_count} ({negative_count/total_episodes*100:.1f}%)")
print(f"正負比例: {positive_count}:{negative_count} = 1:{negative_count/positive_count:.2f}" if positive_count > 0 else "N/A")
print("=" * 60)

# 顯示 max_return 的分布
df_episodes = pd.DataFrame(episodes_info)
print("\n[4] Max Return 分布統計:")
print(df_episodes['max_return_pct'].describe())

# 顯示各區間的分布
print("\n[5] Max Return 區間分布:")
bins = [-100, 0, 5, 10, 15, 20, 30, 50, 100, 200]
labels = ['< 0%', '0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30-50%', '50-100%', '> 100%']
df_episodes['return_bin'] = pd.cut(df_episodes['max_return_pct'], bins=bins, labels=labels)
print(df_episodes['return_bin'].value_counts().sort_index())
