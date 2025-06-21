import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.utils.db.mysql_engine import MysqlEngine


# ========== 第二部分：数据处理 ==========
def clean_data(df):
    # 基础清洗
    df = df.dropna(subset=['close', 'vol'])
    df = df[df['vol'] > 0]  # 剔除停牌数据

    # 添加必要字段
    df['pct_chg'] = df.groupby('ts_code')['close'].pct_change()
    return df


# ========== 第三部分：策略逻辑 ==========
def apply_strategy(df):
    # 重置索引确保所有数据在列中
    df = df.reset_index(drop=True)

    # 计算技术指标（添加参数保持索引对齐）
    def calculate_tech(df_group):
        df_group = df_group.sort_values('trade_date').reset_index(drop=True)
        df_group['ma5'] = df_group['close'].rolling(5).mean()
        df_group['ma20'] = df_group['close'].rolling(20).mean()
        df_group['momentum_20d'] = df_group['close'].pct_change(20)
        return df_group

    # 修改分组方式避免索引冲突
    df = df.groupby('ts_code', group_keys=False).apply(calculate_tech)

    # 生成信号（修正索引问题）
    latest_date = df['trade_date'].max()
    latest_data = df[df['trade_date'] == latest_date].copy()

    # 计算动量排名（确保使用正确的数据）
    latest_data['momentum_rank'] = latest_data.groupby('ts_code')['momentum_20d'].rank(pct=True)

    # 合并排名数据到主表
    df = pd.merge(df, latest_data[['ts_code', 'momentum_rank']], on='ts_code', how='left', validate="many_to_many")

    # 生成信号条件
    condition1 = (df['ma5'] > df['ma20']) & (df['ma5'].shift(1) <= df['ma20'].shift(1))
    condition2 = df['momentum_rank'] > 0.8
    df['signal'] = np.where(condition1 & condition2, 1, 0)

    return df


# ========== 第四部分：执行筛选 ==========
def select_stocks(df):
    # 获取最新交易日数据
    latest_date = df['trade_date'].max()
    latest_df = df[df['trade_date'] == latest_date]

    # 筛选信号股票
    selected = latest_df[latest_df['signal'] == 1]
    return selected['ts_code'].tolist()


# ========== 第五部分：简单回测 ==========
def backtest(df, selected_stocks):
    # 构建等权组合
    portfolio = df[df['ts_code'].isin(selected_stocks)]
    portfolio = portfolio.groupby('trade_date')['pct_chg'].mean().reset_index()

    # 计算累计收益
    portfolio['cum_return'] = (1 + portfolio['pct_chg']).cumprod()

    # 可视化
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=portfolio, x='trade_date', y='cum_return')
    plt.title('Portfolio Cumulative Return')
    plt.show()


# ========== 主程序 ==========
if __name__ == "__main__":
    # # 步骤1：获取数据
    sql = "select * from  stock_daily where trade_date >= '2024-01-01 00:00:00'"
    df = MysqlEngine.query_to_dataframe(sql)
    df = df.sort_values('trade_date')

    # 步骤2：数据清洗
    print("数据清洗中...")
    cleaned_data = clean_data(df)

    # 步骤3：应用策略
    print("应用策略逻辑...")
    strategy_data = apply_strategy(cleaned_data)

    # 步骤4：执行筛选
    print("筛选符合条件股票...")
    selected = select_stocks(strategy_data)
    print(f"选中股票列表：{selected}")

    # 步骤5：回测
    if len(selected) > 0:
        print("进行策略回测...")
        backtest(strategy_data, selected)
    else:
        print("本次未筛选出符合条件的股票")
