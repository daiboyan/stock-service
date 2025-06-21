"""
A股量化选股系统 - 结构化版本
包含：数据管理、因子工程、因子分析、回测引擎、机器学习模型、结果可视化六大模块
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
import os

from core.stock.ai_chat import DeepSeekChat
from core.utils.db.mysql_engine import MysqlEngine

warnings.filterwarnings('ignore')

os.environ['LDFLAGS'] = '-L/usr/local/opt/libomp/lib'
os.environ['CPPFLAGS'] = '-I/usr/local/opt/libomp/include'


# ======================
# 1. 数据管理模块
# ======================
class DataManager:
    """数据加载、清洗和预处理"""

    def __init__(self):
        self.data = None
        self.clean_data = None

    def load_data(self):
        """加载原始数据"""
        print("正在加载数据...")
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
                 where sd.trade_date >= '2024-01-01 00:00:00' \
              """
        self.data = MysqlEngine.query_to_dataframe(sql)
        print(f"数据加载完成，共 {len(self.data):,} 条记录")
        return self.data

    def clean_and_preprocess(self):
        """数据清洗和预处理"""
        if self.data is None:
            raise ValueError("请先加载数据")

        print("数据清洗中...")
        df = self.data.copy()

        # 基础清洗
        df = df.sort_values(['ts_code', 'trade_date'])
        df = df.drop_duplicates(['ts_code', 'trade_date'])

        # 添加基本特征
        df['returns'] = df.groupby('ts_code')['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])

        # 处理异常值和缺失值
        df = df[df['volume'] > 0]  # 去除停牌数据
        df['volume'] = np.log1p(df['volume'])  # 对成交量取对数

        # 内存优化
        df['ts_code'] = df['ts_code'].astype('category')
        float_cols = ['open', 'high', 'low', 'close', 'amount']
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast='float')

        self.clean_data = df
        print("数据清洗完成")
        return self.clean_data

    def get_clean_data(self):
        """获取清洗后的数据"""
        if self.clean_data is None:
            self.clean_and_preprocess()
        return self.clean_data.copy()


# ======================
# 2. 因子工程模块
# ======================
class FactorEngine:
    """因子计算和管理"""

    def __init__(self, data):
        self.data = data
        self.factor_list = []

    def calculate_momentum_factors(self):
        """计算动量类因子"""
        print("计算动量因子...")
        df = self.data.copy()

        # 动量因子
        df['momentum_1m'] = df.groupby('ts_code')['close'].pct_change(21)
        df['momentum_3m'] = df.groupby('ts_code')['close'].pct_change(63)
        df['momentum_6m'] = df.groupby('ts_code')['close'].pct_change(126)
        df['momentum_12m'] = df.groupby('ts_code')['close'].pct_change(252)

        self.factor_list.extend(['momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m'])
        self.data = df
        return df

    def calculate_volatility_factors(self):
        """计算波动率因子"""
        print("计算波动率因子...")
        df = self.data.copy()

        # 波动率因子
        df['volatility_10d'] = df.groupby('ts_code')['returns'].rolling(10).std().reset_index(0, drop=True)
        df['volatility_30d'] = df.groupby('ts_code')['returns'].rolling(30).std().reset_index(0, drop=True)

        self.factor_list.extend(['volatility_10d', 'volatility_30d'])
        self.data = df
        return df

    def calculate_technical_indicators(self):
        """计算技术指标因子"""
        print("计算技术指标因子...")
        df = self.data.copy()

        # RSI
        def calculate_rsi(group, window=14):
            close = group['close']
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        df['rsi'] = df.groupby('ts_code').apply(calculate_rsi).reset_index(0, drop=True)

        # MACD
        def calculate_macd(group, fast=12, slow=26, signal=9):
            close = group['close']
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return macd - signal_line

        df['macd'] = df.groupby('ts_code').apply(calculate_macd).reset_index(0, drop=True)

        # Bollinger Bands
        def calculate_bollinger(group, window=20):
            close = group['close']
            sma = close.rolling(window).mean()
            std = close.rolling(window).std()
            return (close - sma) / (2 * std)

        df['bollinger'] = df.groupby('ts_code').apply(calculate_bollinger).reset_index(0, drop=True)

        self.factor_list.extend(['rsi', 'macd', 'bollinger'])
        self.data = df
        return df

    def calculate_volume_factors(self):
        """计算成交量因子"""
        print("计算成交量因子...")
        df = self.data.copy()

        # 成交量因子
        df['volume_ma_ratio'] = df.groupby('ts_code')['volume'].transform(
            lambda x: x / x.rolling(30).mean())

        self.factor_list.append('volume_ma_ratio')
        self.data = df
        return df

    def calculate_all_factors(self):
        """计算所有因子"""
        self.calculate_momentum_factors()
        self.calculate_volatility_factors()
        self.calculate_technical_indicators()
        self.calculate_volume_factors()
        print(f"因子计算完成，共 {len(self.factor_list)} 个因子")
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
        self.ic_results = {}

    def calculate_factor_ic(self, forward_window=5):
        """计算因子IC值"""
        print("\n计算因子IC值...")

        # 创建远期收益率
        self.data[f'forward_{forward_window}d'] = self.data.groupby('ts_code')['close'].pct_change(
            forward_window).shift(
            -forward_window)

        ic_results = {}
        for factor in tqdm(self.factor_list, desc="因子IC分析"):
            # 按日期分组计算相关系数
            ic_series = self.data.groupby('trade_date').apply(
                lambda x: x[factor].corr(x[f'forward_{forward_window}d']))
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ic_results[factor] = {'IC Mean': ic_mean, 'IC Std': ic_std}

            self.ic_results = ic_results
        return ic_results

    def visualize_ic_results(self):
        """可视化IC分析结果"""
        if not self.ic_results:
            raise ValueError("请先计算因子IC值")

        ic_df = pd.DataFrame(self.ic_results).T
        ic_df = ic_df.sort_values('IC Mean', ascending=False)

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

        return ic_df

    def select_top_factors(self, n=5):
        """选择表现最好的n个因子"""
        if not self.ic_results:
            self.calculate_factor_ic()

        ic_df = pd.DataFrame(self.ic_results).T
        top_factors = ic_df.sort_values('IC Mean', ascending=False).head(n).index.tolist()
        print(f"\n选定的顶级因子: {top_factors}")
        return top_factors


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
        self.performance = None

    def prepare_backtest_data(self, factor_score_col, quantile=1, rebalance_freq='M'):
        """准备回测数据"""
        print("准备回测数据...")
        df = self.data.copy()

        # 标记调仓日
        df['rebalance_day'] = False
        if rebalance_freq == 'M':
            df.loc[df['trade_date'].dt.is_month_end, 'rebalance_day'] = True
        elif rebalance_freq == 'W':
            df.loc[df['trade_date'].dt.dayofweek == 4, 'rebalance_day'] = True

        # 创建持仓矩阵
        df['position'] = 0

        # 处理因子分组 - 添加对NaN值的检查
        def safe_qcut(x):
            """安全的qcut函数，处理全NaN情况"""
            # 如果因子值全为NaN，返回默认值3（中性）
            if x.isnull().all():
                return pd.Series([3] * len(x), index=x.index)

            # 如果有有效值，进行正常分箱
            try:
                return pd.qcut(x, 5, labels=False, duplicates='drop') + 1
            except ValueError:
                # 如果分箱失败（如所有值相同），返回均匀分组
                return pd.Series(np.arange(len(x)) % 5 + 1, index=x.index)

        # 应用安全的分组函数
        df['factor_quantile'] = df.groupby('trade_date')[factor_score_col].transform(safe_qcut)

        # 仅对指定分位数的股票标记为持仓
        df.loc[df['factor_quantile'] == quantile, 'position'] = 1

        # 计算每日收益
        df['strategy_returns'] = df.groupby('ts_code')['returns'].shift(-1)

        return df

    def run_vectorized_backtest(self, factor_score_col='composite_score', quantile=1):
        """运行向量化回测"""
        print("\n运行回测...")
        df = self.prepare_backtest_data(factor_score_col, quantile)

        # 初始化组合
        portfolio = pd.DataFrame(index=df['trade_date'].unique())
        portfolio = portfolio.sort_index()
        portfolio['capital'] = self.initial_capital
        portfolio['value'] = self.initial_capital
        portfolio['cash'] = self.initial_capital
        portfolio['holdings'] = 0.0
        portfolio['n_stocks'] = 0

        # 获取调仓日
        rebalance_dates = df[df['rebalance_day']]['trade_date'].unique()

        # 按调仓日循环
        for i, date in enumerate(tqdm(rebalance_dates, desc="回测进度")):
            # 获取当日选中的股票
            current_day = df[df['trade_date'] == date]
            selected_stocks = current_day[current_day['factor_quantile'] == quantile]['ts_code'].unique()
            num_stocks = len(selected_stocks)

            if num_stocks == 0:
                continue

            # 计算每只股票的权重 (等权重)
            weight_per_stock = 1.0 / num_stocks

            # 计算需要投资的金额
            invest_amount = portfolio.loc[date, 'cash'] * (1 - self.trade_cost)

            # 更新持仓
            portfolio.loc[date, 'holdings'] = invest_amount
            portfolio.loc[date, 'cash'] = portfolio.loc[date, 'cash'] - invest_amount
            portfolio.loc[date, 'n_stocks'] = num_stocks

            # 确定下一个调仓日
            next_rebalance = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else df['trade_date'].max()

            # 计算期间收益
            period_data = df[
                (df['trade_date'] > date) &
                (df['trade_date'] <= next_rebalance) &
                (df['ts_code'].isin(selected_stocks))]

            if period_data.empty:
                continue

            # 计算组合每日收益
            daily_returns = period_data.groupby('trade_date')['strategy_returns'].mean()

            # 更新组合价值
            current_value = invest_amount
            for day in daily_returns.index:
                if day not in portfolio.index:
                    continue

                # 计算当日持仓价值
                current_value *= (1 + daily_returns[day])
                portfolio.loc[day, 'holdings'] = current_value
                portfolio.loc[day, 'value'] = current_value + portfolio.loc[date, 'cash']
                portfolio.loc[day, 'n_stocks'] = num_stocks

        # 填充缺失值
        portfolio = portfolio.fillna(method='ffill')
        portfolio['returns'] = portfolio['value'].pct_change()
        self.portfolio = portfolio
        return portfolio

    def evaluate_performance(self):
        """评估策略表现"""
        if self.portfolio is None:
            raise ValueError("请先运行回测")

        returns = self.portfolio['returns'].dropna()
        cum_returns = (1 + returns).cumprod()

        # 计算关键指标
        annual_return = cum_returns.iloc[-1] ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0

        # 计算最大回撤
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min()

        # 保存结果
        self.performance = {
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cum_returns
        }

        return self.performance

    def visualize_performance(self, benchmark=None):
        """可视化策略表现"""
        if self.performance is None:
            self.evaluate_performance()

        cum_returns = self.performance['cumulative_returns']

        plt.figure(figsize=(14, 7))

        # 绘制策略净值
        plt.plot(cum_returns, label='策略净值', linewidth=2)

        # 绘制基准（如果有）
        if benchmark is not None:
            plt.plot(benchmark, label='基准', linestyle='--')

        # 绘制最大回撤
        peak = cum_returns.cummax()
        plt.fill_between(cum_returns.index, cum_returns, peak,
                         where=(cum_returns < peak), color='red', alpha=0.3,
                         label='回撤区域')

        # 添加标注
        plt.annotate(f'年化收益: {self.performance["annual_return"]:.2%}\n'
                     f'夏普比率: {self.performance["sharpe_ratio"]:.2f}\n'
                     f'最大回撤: {self.performance["max_drawdown"]:.2%}',
                     xy=(0.05, 0.7), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.title('策略表现分析')
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
class StockSelectionModel:
    """机器学习选股模型"""

    def __init__(self, data, feature_cols):
        self.data = data
        self.feature_cols = feature_cols
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importances = None

    def prepare_training_data(self, horizon=5, threshold_quantile=0.7):
        """准备训练数据（修复行业特征问题）"""
        print("准备机器学习训练数据...")
        df = self.data.copy()

        # 创建标签
        df['future_returns'] = df.groupby('ts_code')['returns'].shift(-horizon)
        threshold = df['future_returns'].quantile(threshold_quantile)
        df['target'] = (df['future_returns'] > threshold).astype(int)

        # 仅对数值型特征创建滞后特征
        numeric_features = [f for f in self.feature_cols if not f.startswith('ind_')]
        lag_periods = [1, 5, 10]
        for lag in lag_periods:
            for feature in numeric_features:
                df[f'{feature}_lag{lag}'] = df.groupby('ts_code')[feature].shift(lag)

        # 处理行业特征（如果存在）
        if 'industry' in df.columns:
            # 获取所有可能的行业类别（确保训练和预测时一致）
            all_industries = ['IT设备', '医药', '电子', ...]  # 这里替换为实际的行业列表

            # 创建虚拟变量（确保包含所有可能的行业）
            for industry in all_industries:
                col_name = f'ind_{industry}'
                df[col_name] = (df['industry'] == industry).astype(int)

                # 如果是新特征，添加到特征列表
                if col_name not in self.feature_cols:
                    self.feature_cols.append(col_name)

        # 删除缺失值
        df = df.dropna(subset=['target'] + self.feature_cols)

        return df

    def train_xgboost_model(self, n_splits=5):
        """训练XGBoost模型（使用时间序列交叉验证）"""
        print("训练XGBoost模型...")
        ml_df = self.prepare_training_data()

        X = ml_df[self.feature_cols]
        y = ml_df['target']

        # 初始化模型
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=n_splits)
        feature_importances = pd.DataFrame(index=self.feature_cols)
        metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"\n交叉验证 Fold {fold + 1}/{n_splits}")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 标准化特征
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # 训练模型
            self.model.fit(X_train_scaled, y_train)

            # 预测
            y_pred = self.model.predict(X_test_scaled)

            # 评估
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics.append({
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score'],
                'accuracy': report['accuracy']
            })

            # 保存特征重要性
            fold_importances = pd.Series(self.model.feature_importances_,
                                         index=self.feature_cols)
            feature_importances[f'fold_{fold + 1}'] = fold_importances

        # 保存结果
        self.feature_importances = feature_importances
        avg_metrics = pd.DataFrame(metrics).mean()

        print("\n平均模型性能:")
        print(avg_metrics)

        return avg_metrics

    def visualize_feature_importance(self, top_n=15):
        """可视化特征重要性"""
        if self.feature_importances is None:
            raise ValueError("请先训练模型")

        # 计算平均重要性
        feature_importances = self.feature_importances.copy()
        feature_importances['mean'] = feature_importances.mean(axis=1)
        feature_importances = feature_importances.sort_values('mean', ascending=True)

        # 取最重要的top_n个特征
        top_features = feature_importances['mean'].tail(top_n)

        plt.figure(figsize=(10, 8))
        top_features.plot(kind='barh', color='skyblue')
        plt.title('Top 特征重要性')
        plt.xlabel('重要性得分')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()

    def predict_stock_selection(self):
        try:
            if self.model is None:
                self.train_xgboost_model()

            ml_df = self.prepare_training_data()
            # 确保只使用模型训练时存在的特征
            available_features = [f for f in self.feature_cols if f in ml_df.columns]
            X = ml_df[available_features]

            X_scaled = self.scaler.transform(X)
            ml_df['ml_score'] = self.model.predict_proba(X_scaled)[:, 1]
            return ml_df
        except Exception as e:
            print(f"预测失败: {str(e)}")
            return None


# ========== 6. 生成最终选股结果 ==========
def get_current_recommendations(ml_data, top_n=20):
    """获取最新一期的推荐股票"""
    print("\n生成当前推荐股票列表...")

    # 确保日期排序
    ml_data = ml_data.sort_values('trade_date')

    # 获取最新数据日期
    latest_date = ml_data['trade_date'].max()
    latest_data = ml_data[ml_data['trade_date'] == latest_date]

    # 按综合评分选股
    recommendations = latest_data.nlargest(top_n, 'composite_score')  # 也可以使用ml_score

    # 添加必要信息
    recommendations = recommendations[[
        'ts_code', 'trade_date', 'close',
        'composite_score', 'ml_score', 'industry'
    ]].copy()
    recommendations['weight'] = 1 / top_n  # 等权重配置

    print(recommendations['ts_code'])

    return recommendations


# ======================
# 6. 主程序
# ======================
def main():
    # 初始化设置
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    plt.style.use(PLOT_STYLE)

    # ========== 1. 数据管理 ==========
    print("\n" + "=" * 50)
    print("数据管理阶段")
    print("=" * 50)
    data_manager = DataManager()
    data_manager.load_data()
    clean_data = data_manager.clean_and_preprocess()

    # ========== 2. 因子工程 ==========
    print("\n" + "=" * 50)
    print("因子工程阶段")
    print("=" * 50)
    factor_engine = FactorEngine(clean_data)
    factor_data = factor_engine.calculate_all_factors()
    factor_list = factor_engine.get_factor_list()

    # ========== 3. 因子分析 ==========
    print("\n" + "=" * 50)
    print("因子分析阶段")
    print("=" * 50)
    factor_analyzer = FactorAnalyzer(factor_data, factor_list)
    ic_results = factor_analyzer.calculate_factor_ic()
    ic_df = factor_analyzer.visualize_ic_results()
    selected_factors = factor_analyzer.select_top_factors(n=5)

    # 构建复合因子
    print("\n构建复合因子...")
    for factor in selected_factors:
        factor_data[f'{factor}_z'] = factor_data.groupby('trade_date')[factor].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )

    # 等权复合因子
    z_columns = [f'{f}_z' for f in selected_factors]
    factor_data['composite_score'] = factor_data[z_columns].mean(axis=1)

    # ========== 4. 回测引擎 ==========
    print("\n" + "=" * 50)
    print("回测阶段")
    print("=" * 50)
    backtester = BacktestEngine(factor_data)
    portfolio = backtester.run_vectorized_backtest(factor_score_col='composite_score')
    performance = backtester.evaluate_performance()
    print("\n策略表现:")
    for k, v in performance.items():
        if k != 'cumulative_returns':
            print(f"{k.replace('_', ' ').title()}: {v:.4f}")
    backtester.visualize_performance()

    # ========== 5. 机器学习模型 ==========
    print("\n" + "=" * 50)
    print("机器学习选股阶段")
    print("=" * 50)
    ml_model = StockSelectionModel(factor_data, selected_factors)
    ml_model.train_xgboost_model()
    ml_model.visualize_feature_importance()

    # 生成选股结果
    print("\n生成选股结果...")
    ml_results = ml_model.predict_stock_selection()

    # 获取并保存推荐股票
    current_picks = get_current_recommendations(ml_results, top_n=15)
    print("\n=== 当前推荐股票 ===")
    print(current_picks[['ts_code', 'close', 'composite_score', 'industry']])

    # 保存结果
    current_picks.to_csv('current_stock_picks.csv', index=False)
    print(f"\n最新推荐股票已保存，共 {len(current_picks)} 只")


def ai_analysis():
    chat_bot = DeepSeekChat()
    df = pd.read_csv("/Users/daibingbing/PyCharmMiscProject/core/stock/current_stock_picks.csv", encoding="utf-8")
    stock_codes = "、".join(df['ts_code'])
    promot = f"{stock_codes}这些是我量化选出来的股票，帮我分析是否值得投资，要求格式为：股票编码（股票名称）：值得投资的星级（最高10星）,投资建议，简洁清晰。"
    print(chat_bot.chat_online(promot))


def main2():
    # 初始化设置
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    plt.style.use(PLOT_STYLE)

    # ========== 1. 数据管理 ==========
    print("\n" + "=" * 50)
    print("数据管理阶段")
    print("=" * 50)
    data_manager = DataManager()

    # 只加载最新数据（减少计算量）
    data_manager.load_data()
    clean_data = data_manager.clean_and_preprocess()

    # 获取最新交易日
    latest_date = clean_data['trade_date'].max()
    print(f"最新交易日: {latest_date}")

    # ========== 2. 因子工程 ==========
    print("\n" + "=" * 50)
    print("因子工程阶段")
    print("=" * 50)
    factor_engine = FactorEngine(clean_data)

    # 优化：只计算最新交易日需要的因子
    latest_data = clean_data[clean_data['trade_date'] == latest_date].copy()
    factor_engine.calculate_all_factors()

    # 获取因子列表
    factor_list = factor_engine.get_factor_list()

    # ========== 3. 因子分析 ==========
    print("\n" + "=" * 50)
    print("因子分析阶段")
    print("=" * 50)
    FactorAnalyzer(clean_data, factor_list)

    # 简化：直接选择重要因子，不计算历史IC
    selected_factors = [
        'momentum_3m', 'volatility_30d',
        'rsi', 'macd', 'volume_ma_ratio'
    ]
    print(f"选定的顶级因子: {selected_factors}")

    # 构建复合因子（仅最新数据）
    print("\n构建复合因子...")
    for factor in selected_factors:
        latest_data[f'{factor}_z'] = (latest_data[factor] - latest_data[factor].mean()) / latest_data[factor].std()

    z_columns = [f'{f}_z' for f in selected_factors]
    latest_data['composite_score'] = latest_data[z_columns].mean(axis=1)

    # ========== 4. 机器学习选股 ==========
    print("\n" + "=" * 50)
    print("机器学习选股阶段")
    print("=" * 50)

    # 准备训练数据（使用全量历史数据）
    ml_model = StockSelectionModel(clean_data, selected_factors)
    ml_model.train_xgboost_model()

    # 只预测最新交易日的股票
    ml_model.data = latest_data  # 使用最新数据
    latest_ml_results = ml_model.predict_stock_selection()

    # 添加机器学习评分到最新数据
    latest_data = latest_data.merge(
        latest_ml_results[['ts_code', 'trade_date', 'ml_score']],
        on=['ts_code', 'trade_date'],
        how='left'
    )

    # 组合评分 = 因子评分 + ML评分
    latest_data['final_score'] = 0.7 * latest_data['composite_score'] + 0.3 * latest_data['ml_score']

    # ========== 5. 生成选股结果 ==========
    print("\n" + "=" * 50)
    print("生成选股结果")
    print("=" * 50)

    # 获取并保存推荐股票
    current_picks = get_current_recommendations(latest_data, top_n=15)

    print("\n=== 当前推荐股票 ===")
    print(current_picks[['ts_code', 'close', 'final_score', 'industry']])

    # 保存结果
    current_picks.to_csv('current_stock_picks.csv', index=False)
    print(f"\n最新推荐股票已保存，共 {len(current_picks)} 只")



if __name__ == "__main__":
    main()
    ai_analysis()

