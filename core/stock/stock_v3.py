"""
A股量化选股系统 - 优化结构化版本
包含：数据管理、因子工程、因子分析、回测引擎、机器学习模型五大核心模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings

from core.utils.db.mysql_engine import MysqlEngine

warnings.filterwarnings('ignore')


# ======================
# 1. 数据管理模块
# ======================
class DataManager:
    """数据加载、清洗和预处理"""

    def __init__(self):
        self.data = None

    def load_and_clean_data(self):
        """加载并清洗数据"""
        print("正在加载并清洗数据...")

        # 加载数据
        sql = """SELECT sd.trade_date,
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
                 where sd.trade_date >= '2024-01-01 00:00:00' \
              """
        self.data = MysqlEngine.query_to_dataframe(sql)
        print(f"原始数据记录数: {len(self.data):,}")

        # 基础清洗
        df = self.data.sort_values(['code', 'trade_date'])
        df = df.drop_duplicates(['code', 'trade_date'])

        # 添加基本特征
        df['returns'] = df.groupby('code')['close'].pct_change()

        # 处理异常值和缺失值
        df = df[df['volume'] > 0]  # 去除停牌数据
        df = df.dropna(subset=['open', 'high', 'low', 'close'])

        # 内存优化
        df['code'] = df['code'].astype('category')
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], downcast='float')

        print(f"清洗后数据记录数: {len(df):,}")
        self.data = df
        return self.data


# ======================
# 2. 因子工程模块
# ======================
class FactorEngine:
    """因子计算和管理"""

    def __init__(self, data):
        self.data = data
        self.factor_list = []

    def calculate_factors(self):
        """计算所有因子"""
        print("开始计算因子...")
        df = self.data.copy()

        # 动量因子
        df['momentum_1m'] = df.groupby('code')['close'].pct_change(21)
        df['momentum_3m'] = df.groupby('code')['close'].pct_change(63)
        df['momentum_6m'] = df.groupby('code')['close'].pct_change(126)
        df['momentum_12m'] = df.groupby('code')['close'].pct_change(252)
        self.factor_list.extend(['momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m'])

        # 波动率因子
        df['volatility_10d'] = df.groupby('code')['returns'].rolling(10).std().reset_index(0, drop=True)
        df['volatility_30d'] = df.groupby('code')['returns'].rolling(30).std().reset_index(0, drop=True)
        self.factor_list.extend(['volatility_10d', 'volatility_30d'])

        # RSI
        def calculate_rsi(group, window=14):
            close = group['close']
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            rs = avg_gain / (avg_loss + 1e-8)  # 避免除零
            return 100 - (100 / (1 + rs))

        df['rsi'] = df.groupby('code').apply(calculate_rsi).reset_index(0, drop=True)
        self.factor_list.append('rsi')

        # MACD
        def calculate_macd(group, fast=12, slow=26, signal=9):
            close = group['close']
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return macd - signal_line

        df['macd'] = df.groupby('code').apply(calculate_macd).reset_index(0, drop=True)
        self.factor_list.append('macd')

        # 成交量因子
        df['volume_ma_ratio'] = df.groupby('code')['volume'].transform(
            lambda x: x / x.rolling(30).mean())
        self.factor_list.append('volume_ma_ratio')

        print(f"因子计算完成，共 {len(self.factor_list)} 个因子")
        self.data = df
        return self.data

    def get_factor_list(self):
        """获取因子列表"""
        return self.factor_list.copy()


# ======================
# 3. 因子分析模块
# ======================
class FactorAnalyzer:
    """因子分析和选择"""

    def __init__(self, data, factor_list):
        self.data = data
        self.factor_list = factor_list

    def analyze_factors(self, forward_window=5):
        """分析因子并选择最佳因子"""
        print("\n开始因子分析...")

        # 创建远期收益率
        self.data[f'forward_{forward_window}d'] = self.data.groupby('code')['close'].pct_change(forward_window).shift(
            -forward_window)

        # 计算因子IC值
        ic_results = {}
        for factor in tqdm(self.factor_list, desc="计算因子IC值"):
            # 按日期分组计算相关系数
            ic_series = self.data.groupby('trade_date').apply(
                lambda x: x[factor].corr(x[f'forward_{forward_window}d'], method='spearman'))
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ic_ir = ic_mean / ic_std  # IC信息比率
            ic_results[factor] = {'IC Mean': ic_mean, 'IC Std': ic_std, 'IC IR': ic_ir}

        # 转换为DataFrame并排序
        ic_df = pd.DataFrame(ic_results).T
        ic_df = ic_df.sort_values('IC Mean', ascending=False)

        # 可视化结果
        plt.figure(figsize=(12, 6))
        ic_df['IC Mean'].plot(kind='bar', yerr=ic_df['IC Std'],
                              capsize=4, color='skyblue')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('因子IC分析')
        plt.ylabel('信息系数 (IC)')
        plt.xlabel('因子')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('factor_ic_analysis.png')
        plt.show()

        # 选择表现最好的5个因子
        top_factors = ic_df.head(5).index.tolist()
        print(f"\n选定的顶级因子: {top_factors}")

        return top_factors, ic_df

    def build_composite_factor(self, selected_factors):
        """构建复合因子"""
        print("构建复合因子...")
        df = self.data.copy()

        # 因子标准化
        for factor in selected_factors:
            df[f'{factor}_z'] = df.groupby('trade_date')[factor].transform(
                lambda x: (x - x.mean()) / x.std())

        # 等权复合因子
        z_columns = [f'{f}_z' for f in selected_factors]
        df['composite_score'] = df[z_columns].mean(axis=1)

        # 因子分组
        df['factor_quantile'] = df.groupby('trade_date')['composite_score'].transform(
            lambda x: pd.qcut(x, 5, labels=False) + 1)

        self.data = df
        return self.data


# ======================
# 4. 回测引擎模块
# ======================
class BacktestEngine:
    """回测引擎"""

    def __init__(self, data, initial_capital=1000000, trade_cost=0.001):
        self.data = data
        self.initial_capital = initial_capital
        self.trade_cost = trade_cost
        self.portfolio = None

    def run_backtest(self, quantile=1):
        """运行回测"""
        print("\n运行回测...")
        df = self.data.copy()

        # 标记调仓日（每月最后一天）
        df['rebalance_day'] = df['trade_date'].dt.is_month_end

        # 准备回测数据
        df['next_returns'] = df.groupby('code')['returns'].shift(-1)
        df = df.dropna(subset=['next_returns'])

        # 初始化组合
        portfolio = pd.DataFrame(index=df['trade_date'].unique())
        portfolio = portfolio.sort_index()
        portfolio['capital'] = self.initial_capital
        portfolio['cash'] = self.initial_capital
        portfolio['holdings_value'] = 0.0
        portfolio['total_value'] = self.initial_capital
        portfolio['n_stocks'] = 0

        # 获取调仓日
        rebalance_dates = df[df['rebalance_day']]['trade_date'].unique()

        # 按调仓日循环
        for i, date in enumerate(tqdm(rebalance_dates, desc="回测进度")):
            # 获取当日选中的股票
            current_day = df[df['trade_date'] == date]
            selected_stocks = current_day[current_day['factor_quantile'] == quantile]
            num_stocks = len(selected_stocks)

            if num_stocks == 0:
                continue

            # 计算每只股票的权重 (等权重)
            weight_per_stock = 1.0 / num_stocks

            # 计算需要投资的金额
            available_cash = portfolio.loc[date, 'cash']
            invest_amount = available_cash * (1 - self.trade_cost)

            # 更新持仓
            portfolio.loc[date, 'holdings_value'] = invest_amount
            portfolio.loc[date, 'cash'] = available_cash - invest_amount
            portfolio.loc[date, 'n_stocks'] = num_stocks

            # 确定下一个调仓日
            next_rebalance = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else df['trade_date'].max()

            # 计算期间收益
            period_data = df[
                (df['trade_date'] > date) &
                (df['trade_date'] <= next_rebalance) &
                (df['code'].isin(selected_stocks['code']))]

            if period_data.empty:
                continue

            # 计算组合每日收益
            daily_returns = period_data.groupby('trade_date')['next_returns'].mean()

            # 更新组合价值
            current_value = invest_amount
            for day in daily_returns.index:
                if day not in portfolio.index:
                    continue

                # 计算当日持仓价值
                current_value *= (1 + daily_returns[day])
                portfolio.loc[day, 'holdings_value'] = current_value
                portfolio.loc[day, 'total_value'] = current_value + portfolio.loc[date, 'cash']
                portfolio.loc[day, 'n_stocks'] = num_stocks

        # 填充缺失值
        portfolio = portfolio.ffill()
        portfolio['returns'] = portfolio['total_value'].pct_change()
        self.portfolio = portfolio
        return portfolio

    def evaluate_performance(self):
        """评估策略表现"""
        if self.portfolio is None:
            raise ValueError("请先运行回测")

        portfolio = self.portfolio.dropna(subset=['returns'])
        returns = portfolio['returns']
        cum_returns = (1 + returns).cumprod()

        # 计算关键指标
        total_return = cum_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0

        # 计算最大回撤
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min()

        # 输出结果
        print("\n========== 策略表现 ==========")
        print(f"累计收益: {total_return:.2%}")
        print(f"年化收益: {annual_return:.2%}")
        print(f"年化波动: {volatility:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"最大回撤: {max_drawdown:.2%}")

        # 保存结果
        performance = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cum_returns': cum_returns
        }

        return performance

    def plot_performance(self, performance):
        """绘制策略表现图"""
        cum_returns = performance['cum_returns']

        plt.figure(figsize=(12, 6))
        plt.plot(cum_returns, label='策略净值', linewidth=2)

        # 绘制最大回撤
        peak = cum_returns.cummax()
        plt.fill_between(cum_returns.index, cum_returns, peak,
                         where=(cum_returns < peak), color='red', alpha=0.3,
                         label='回撤区域')

        # 添加标注
        plt.annotate(f'年化收益: {performance["annual_return"]:.2%}\n'
                     f'夏普比率: {performance["sharpe_ratio"]:.2f}\n'
                     f'最大回撤: {performance["max_drawdown"]:.2%}',
                     xy=(0.05, 0.7), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.title('策略净值曲线')
        plt.xlabel('日期')
        plt.ylabel('累计收益')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('strategy_performance.png')
        plt.show()


# ======================
# 5. 机器学习模型模块
# ======================
class MLStockSelector:
    """机器学习选股模型"""

    def __init__(self, data, features):
        self.data = data
        self.features = features
        self.model = None
        self.scaler = StandardScaler()

    def prepare_data(self, horizon=5):
        """准备机器学习数据"""
        print("准备机器学习数据...")
        df = self.data.copy()

        # 创建标签 - 未来N日收益率
        df['future_returns'] = df.groupby('code')['returns'].shift(-horizon)

        # 仅保留有未来收益的数据
        df = df.dropna(subset=['future_returns'])

        # 创建二元分类标签 (前30%)
        threshold = df['future_returns'].quantile(0.7)
        df['target'] = (df['future_returns'] > threshold).astype(int)

        # 添加滞后特征
        for feature in self.features:
            df[f'{feature}_lag1'] = df.groupby('code')[feature].shift(1)
            df[f'{feature}_lag5'] = df.groupby('code')[feature].shift(5)

        # 更新特征列表
        new_features = self.features.copy()
        for f in self.features:
            new_features.extend([f'{f}_lag1', f'{f}_lag5'])

        # 删除缺失值
        df = df.dropna(subset=new_features + ['target'])

        return df, new_features

    def train_model(self, n_splits=3):
        """训练XGBoost模型"""
        print("训练机器学习模型...")
        df, features = self.prepare_data()
        X = df[features]
        y = df['target']

        # 初始化模型
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = []

        print(f"使用 {len(features)} 个特征进行 {n_splits} 折交叉验证")
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"\nFold {fold + 1}/{n_splits}")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 标准化特征
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            # 训练模型
            self.model.fit(X_train, y_train)

            # 预测
            y_pred = self.model.predict(X_test)

            # 评估
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics.append(report['accuracy'])
            print(f"准确率: {report['accuracy']:.4f}")

        # 输出平均性能
        avg_accuracy = np.mean(metrics)
        print(f"\n平均准确率: {avg_accuracy:.4f}")

        return avg_accuracy

    def generate_predictions(self):
        """生成预测结果"""
        if self.model is None:
            self.train_model()

        df, features = self.prepare_data()
        X = df[features]

        # 标准化并预测
        X_scaled = self.scaler.transform(X)
        df['ml_score'] = self.model.predict_proba(X_scaled)[:, 1]

        # 保存结果
        result_cols = ['code', 'trade_date', 'close', 'returns', 'future_returns', 'ml_score']
        results = df[result_cols].copy()

        # 添加复合因子
        if 'composite_score' in df.columns:
            results['composite_score'] = df['composite_score']

        return results


# ======================
# 主程序
# ======================
def main():
    # 配置设置
    DATA_PATH = 'a_stock_data.csv'
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    plt.style.use(PLOT_STYLE)

    print("=" * 50)
    print("A股量化选股系统")
    print("=" * 50)

    try:
        # 1. 数据准备
        print("\n>>> 阶段1: 数据准备")
        data_manager = DataManager(DATA_PATH)
        clean_data = data_manager.load_and_clean_data()

        # 2. 因子工程
        print("\n>>> 阶段2: 因子工程")
        factor_engine = FactorEngine(clean_data)
        factor_data = factor_engine.calculate_factors()
        factor_list = factor_engine.get_factor_list()

        # 3. 因子分析
        print("\n>>> 阶段3: 因子分析")
        factor_analyzer = FactorAnalyzer(factor_data, factor_list)
        selected_factors, ic_df = factor_analyzer.analyze_factors()

        # 保存IC结果
        ic_df.to_csv('factor_ic_results.csv')
        print("因子IC结果已保存到 factor_ic_results.csv")

        # 构建复合因子
        factor_data = factor_analyzer.build_composite_factor(selected_factors)

        # 4. 回测
        print("\n>>> 阶段4: 策略回测")
        backtester = BacktestEngine(factor_data)
        portfolio = backtester.run_backtest(quantile=1)  # 选择前20%的股票
        performance = backtester.evaluate_performance()
        backtester.plot_performance(performance)

        # 5. 机器学习选股
        print("\n>>> 阶段5: 机器学习选股")
        ml_selector = MLStockSelector(factor_data, selected_factors)
        stock_predictions = ml_selector.generate_predictions()

        # 保存最终结果
        stock_predictions.to_csv('stock_selection_results.csv', index=False)
        print("\n量化选股完成! 结果已保存到 stock_selection_results.csv")

        # 显示样本结果
        print("\n结果示例:")
        print(stock_predictions.tail(10))

    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()