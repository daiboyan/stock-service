import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import os

from core.utils.db.mysql_engine import MysqlEngine

# 假设df是您从数据库获取的DataFrame
# 这里使用模拟数据（实际使用时请替换为您的真实数据）

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
         where sd.trade_date >= '2025-01-01 00:00:00' \
      """
df = MysqlEngine.query_to_dataframe(sql)
df['trade_date'] = pd.to_datetime(df['trade_date'])

# 1. 数据处理
# 按行业和日期聚合，计算每日行业总交易额
industry_daily = df.groupby(['industry', 'trade_date'], as_index=False)['amount'].sum()

# 计算环比变化
industry_daily = industry_daily.sort_values(['industry', 'trade_date'])
industry_daily['prev_amount'] = industry_daily.groupby('industry')['amount'].shift(1)
industry_daily['amount_pct_change'] = (
                                              (industry_daily['amount'] - industry_daily['prev_amount']) /
                                              industry_daily['prev_amount']
                                      ) * 100

# 清理中间列
result = industry_daily.drop(columns='prev_amount').sort_values(['industry', 'trade_date'])

# 2. 可视化设置 - 改进字体设置
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
sns.set_style("whitegrid")

# Mac系统中文字体路径
mac_font_paths = [
    "/System/Library/Fonts/PingFang.ttc",  # PingFang SC
    "/System/Library/Fonts/STHeiti Medium.ttc",  # Heiti SC
    "/System/Library/Fonts/STHeiti Light.ttc"   # Heiti Light
]

# 尝试加载字体
font_found = False
for font_path in mac_font_paths:
    if os.path.exists(font_path):
        try:
            # 创建字体属性
            my_font = fm.FontProperties(fname=font_path)
            # 设置全局字体
            plt.rcParams['font.family'] = my_font.get_name()
            print(f"成功加载字体: {font_path}")
            font_found = True
            break
        except Exception as e:
            print(f"加载字体 {font_path} 失败: {e}")

if not font_found:
    print("警告: 未找到合适的中文字体，使用默认字体")

# 3. 方案一：多行业环比变化对比图
def plot_industry_comparison(data, top_n=5):
    """展示交易量最大的几个行业的环比变化趋势"""
    # 选取交易量最大的几个行业
    top_industries = data.groupby('industry')['amount'].sum().nlargest(top_n).index
    filtered_data = data[data['industry'].isin(top_industries)]

    plt.figure(figsize=(14, 8))
    ax = sns.lineplot(
        data=filtered_data,
        x='trade_date',
        y='amount_pct_change',
        hue='industry',
        style='industry',
        markers=True,
        dashes=False,
        linewidth=2.5
    )

    # 美化图表
    plt.title(f'TOP {top_n} 行业交易额环比变化趋势 (2025年)', fontsize=16)
    plt.xlabel('交易日期', fontsize=12)
    plt.ylabel('环比变化 (%)', fontsize=12)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)  # 添加零线

    # 日期格式化
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    # 添加特殊点标记（最大/最小值）
    for industry in top_industries:
        industry_data = filtered_data[filtered_data['industry'] == industry]
        if not industry_data.empty:
            max_idx = industry_data['amount_pct_change'].idxmax()
            min_idx = industry_data['amount_pct_change'].idxmin()
            max_point = industry_data.loc[max_idx]
            min_point = industry_data.loc[min_idx]

            plt.annotate(f'峰值: {max_point["amount_pct_change"]:.1f}%',
                         (max_point['trade_date'], max_point['amount_pct_change']),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
            plt.annotate(f'谷值: {min_point["amount_pct_change"]:.1f}%',
                         (min_point['trade_date'], min_point['amount_pct_change']),
                         textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)

    plt.legend(title='行业', loc='upper left')
    plt.tight_layout()
    plt.savefig('行业交易量环比对比.png', dpi=300)
    plt.show()


# 4. 方案二：单行业详细变化图
def plot_single_industry(industry_name, data):
    """展示单个行业的详细变化情况"""
    industry_data = data[data['industry'] == industry_name].copy()

    if industry_data.empty:
        print(f"未找到行业: {industry_name}")
        return

    # 创建双Y轴图表
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # 交易额柱状图（主Y轴）
    color = 'tab:blue'
    ax1.set_xlabel('交易日期')
    ax1.set_ylabel('交易额 (元)', color=color)
    ax1.bar(industry_data['trade_date'], industry_data['amount'],
            color=color, alpha=0.6, width=0.8)
    ax1.tick_params(axis='y', labelcolor=color)

    # 环比变化折线图（次Y轴）
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('环比变化 (%)', color=color)
    ax2.plot(industry_data['trade_date'], industry_data['amount_pct_change'],
             color=color, marker='o', linewidth=2.5, markersize=6)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)

    # 添加特殊点标签
    max_idx = industry_data['amount_pct_change'].idxmax()
    min_idx = industry_data['amount_pct_change'].idxmin()
    max_point = industry_data.loc[max_idx]
    min_point = industry_data.loc[min_idx]

    ax2.annotate(f'最大涨幅: {max_point["amount_pct_change"]:.1f}%',
                 (max_point['trade_date'], max_point['amount_pct_change']),
                 textcoords="offset points", xytext=(0, 15), ha='center', fontsize=10,
                 arrowprops=dict(arrowstyle="->", color='red'))
    ax2.annotate(f'最大跌幅: {min_point["amount_pct_change"]:.1f}%',
                 (min_point['trade_date'], min_point['amount_pct_change']),
                 textcoords="offset points", xytext=(0, -20), ha='center', fontsize=10,
                 arrowprops=dict(arrowstyle="->", color='green'))

    # 美化图表
    plt.title(f'{industry_name}行业交易额与环比变化趋势', fontsize=16)
    fig.tight_layout()
    plt.savefig(f'{industry_name}行业交易分析.png', dpi=300)
    plt.show()


if __name__ == '__main__':

    # 5. 执行可视化（现在与前面代码连贯）
    print("数据处理完成，开始生成可视化图表...")

    # 查看交易量最大的行业对比
    plot_industry_comparison(result, top_n=min(5, len(result['industry'].unique())))  # 确保不超过实际行业数

    # 查看特定行业的详细情况
    print("\n可用行业列表:", result['industry'].unique())

    # 选择您感兴趣的行业（这里以第一个行业为例）
    # selected_industry = result['industry'].iloc[0]
    # print(f"\n正在生成 {selected_industry} 行业的详细图表...")
    # plot_single_industry(selected_industry, result)
    #
    # print("\n可视化完成！图表已保存为PNG文件。")