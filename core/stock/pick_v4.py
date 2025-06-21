"""
A股量化选股系统 - 实战简化版
输入：历史日线数据 → 输出：当前推荐股票列表
"""
import pandas as pd
import numpy as np
from datetime import datetime

from core.utils.db.mysql_engine import MysqlEngine


# ======================
# 1. 数据加载与清洗
# ======================
class DataLoader:
    @staticmethod
    def load_and_clean():
        """加载并预处理数据"""
        # 这里替换为您的实际数据加载逻辑
        sql = """SELECT sd.ts_code,
                        sd.trade_date,
                        sd.open,
                        sd.high,
                        sd.low,
                        sd.`close`,
                        sd.pre_close,
                        sd.`change`,
                        sd.pct_chg,
                        sd.vol as volume,
                        sd.amount,
                        sbi.industry
                 from stock_daily sd
                          left join stock_basic_info sbi on
                     sd.ts_code = sbi.ts_code
                 where sd.trade_date >= '2025-01-01 00:00:00'
              """
        df = MysqlEngine.query_to_dataframe(sql)
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # 基础清洗
        df = df[df['volume'] > 0].copy()
        df['returns'] = df.groupby('ts_code')['close'].pct_change()
        return df.sort_values(['ts_code', 'trade_date'])


# ======================
# 2. 因子计算引擎
# ======================
class FactorGenerator:
    @staticmethod
    def calculate_factors(df):
        """计算核心因子"""
        # 动量因子
        df['momentum_1m'] = df.groupby('ts_code')['close'].pct_change(21)

        # 波动率因子
        df['volatility_30d'] = df.groupby('ts_code')['returns'].rolling(30).std().reset_index(0, drop=True)

        # 技术指标（简化版）
        df['rsi'] = df.groupby('ts_code')['close'].transform(
            lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).rolling(14).mean() /
                                         x.diff().clip(upper=0).abs().rolling(14).mean()))))
        return df.dropna()


# ======================
# 3. 股票推荐引擎
# ======================
class StockRecommender:
    def __init__(self, top_n=15):
        self.top_n = top_n
        self.factor_weights = {
            'momentum_1m': 0.4,
            'volatility_30d': -0.3,  # 波动率取负向
            'rsi': 0.3
        }

    def generate_recommendations(self, df):
        """生成当前推荐股票"""
        # 获取最新数据
        latest_date = df['trade_date'].max()
        current_data = df[df['trade_date'] == latest_date].copy()

        # 计算综合评分
        current_data['composite_score'] = sum(
            current_data[factor] * weight
            for factor, weight in self.factor_weights.items()
        )

        # 添加风控过滤
        current_data = current_data[
            (current_data['volatility_30d'] < 0.4) &  # 过滤高波动
            (current_data['volume'] > 1e8)  # 过滤低流动性
            ]

        # 返回推荐股票
        return current_data.nlargest(self.top_n, 'composite_score')[
            ['ts_code', 'trade_date', 'close', 'industry', 'composite_score']
        ].assign(
            weight=lambda x: x['composite_score'] / x['composite_score'].sum()  # 智能权重
        )


# ======================
# 主执行流程
# ======================
if __name__ == "__main__":
    # 1. 加载数据
    print("正在加载数据...")
    data = DataLoader.load_and_clean()

    # 2. 计算因子
    print("计算因子...")
    factor_data = FactorGenerator.calculate_factors(data)

    # 3. 生成推荐
    print("生成推荐股票...")
    recommender = StockRecommender(top_n=15)
    recommendations = recommender.generate_recommendations(factor_data)

    # 4. 输出结果
    print("\n=== 最新推荐股票 ===")
    print(recommendations.to_string(index=False))

    # 保存结果
    recommendations.to_csv(f'stock_recommendations_{datetime.today().date()}.csv', index=False)
    print("\n推荐结果已保存！")