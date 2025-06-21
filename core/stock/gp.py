import pandas as pd
import tushare
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from excel.csv_utils import CsvUtils


class Gp:
    def __init__(self):
        self.pro = tushare.pro_api("d03158c0aa9704114b3114aeee70ab0b160b3648d01a73f88db9d23d")

    def get_gp_basic(self):
        return self.pro.stock_basic(exchange='', list_status='L',
                                    fields='ts_code,symbol,name,area,industry,list_date,market,exchange')

    def get_daily(self, ts_code, trade_date):
        df = self.pro.daily(ts_code=ts_code, trade_date=trade_date)
        return df

    def choose(self):
        pro = self.pro
        """ 数据获取与清洗"""
        # 获取沪深300成分股列表（2023年）
        hs300 = pro.index_weight(index_code='000300.SH', start_date='20230101', end_date='20231231')
        stock_list = hs300['con_code'].unique().tolist()

        # 批量获取股票日线数据
        data_all = []
        for code in stock_list:
            dy = pro.daily(ts_code=code, start_date='20200101', end_date='20231231')
            data_all.append(dy)
        df = pd.concat(data_all)

        """因子选择"""
        # 计算因子与未来收益的相关系数（IC值）
        factor = 'pe_ttm'  # 待测试因子（市盈率）
        df['next_ret'] = df.groupby('ts_code')['close'].pct_change(5).shift(-5)  # 未来5日收益率
        ic = df.groupby('date').apply(lambda x: x[factor].corr(x['next_ret'], method='spearman'))
        print(f"IC均值：{ic.mean():.3f}，IC胜率：{(ic > 0).mean():.2%}")

        """因子打分法:因子标准化 → 加权求和 → 排序选股"""
        # 因子标准化（Z-score）
        factors = ['pe_ttm', 'roe', 'revenue_growth']
        df[factors] = df.groupby('date')[factors].apply(lambda x: (x - x.mean()) / x.std())

        # 加权打分（假设权重：价值30%，质量40%，动量30%）
        df['score'] = 0.3 * df['pe_ttm'] + 0.4 * df['roe'] + 0.3 * df['revenue_growth']

        """机器模型学习"""
        # 特征与标签
        features = ['pe_ttm', 'roe', 'volume_20d_ma']
        target = 'next_5d_ret'  # 未来5日收益率

        # 训练模型
        X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2)
        model = LGBMRegressor()
        model.fit(X_train, y_train)

        # 预测选股
        df['pred_ret'] = model.predict(df[features])
        top_stocks = df.groupby('date').apply(lambda x: x.nlargest(10, 'pred_ret'))


if __name__ == '__main__':
    gp = Gp()
    data = gp.get_daily('002031', '20250519')
    CsvUtils.save_to_csv(data, 'gp.csv')
    # schedule.every(1).minutes.do(gp.get_gp)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)