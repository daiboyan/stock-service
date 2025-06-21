import time
from datetime import datetime, timedelta

import pandas as pd
import tushare as ts
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from core.utils.db.mysql_engine import MysqlEngine

# 设置Tushare Token（需前往tushare.pro官网注册获取）
TOKEN = 'd03158c0aa9704114b3114aeee70ab0b160b3648d01a73f88db9d23d'  # 替换为你的实际token
ts.set_token(TOKEN)
pro = ts.pro_api()


def save_to_database(df: pd.DataFrame, table_name: str):
    df.to_sql(
        name=table_name,  # 表名
        con=MysqlEngine.create_engine(),
        if_exists="append",  # 表存在时追加数据；可选 "replace"（覆盖）或 "fail"（报错）
        index=False,  # 不写入 DataFrame 的索引
        chunksize=1000  # 分块写入提升性能
    )


def save_gp_basic():
    # 获取全量股票基础信息
    all_stocks = pro.stock_basic(exchange='', list_status='L',
                                 fields='ts_code,symbol,name,area,industry,list_date,market,exchange')

    # 查询已存在的股票代码
    exists_df = MysqlEngine.query_to_dataframe("SELECT ts_code FROM stock_basic_info")
    exists_codes = set(exists_df['ts_code']) if not exists_df.empty else set()

    # 筛选新增股票代码
    new_stocks = all_stocks[~all_stocks['ts_code'].isin(exists_codes)]
    print(f"新票{new_stocks}")

    # 批量保存新增股票
    if not new_stocks.empty:
        save_to_database(new_stocks, 'stock_basic_info')


def save_stock_daily(stock_list, start_date, end_date):
    # 开始结束时间，默认为近1年
    start_date = start_date if start_date else (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')  # 1年数据
    end_date = end_date if end_date else datetime.now().strftime('%Y%m%d')

    # 已存在数据库的票
    sql = f"select ts_code from stock_daily sbi group by ts_code"
    print(sql)
    df = MysqlEngine.query_to_dataframe(sql)
    ts_codes = df['ts_code'].unique().tolist()
    # 去重
    new_list = list(filter(lambda x: x not in ts_codes, stock_list))

    times = 1
    for code in new_list:
        # 查询接口
        df = pro.daily(
            ts_code=code,
            start_date=start_date,
            end_date=end_date)
        if not df.empty:
            # 将 DataFrame 写入数据库表（自动创建表或追加数据）
            print(f"【{code}】已经开始")
            print(times)
            try:
                save_to_database(df, "stock_daily")
            except Exception as e:
                print(e)
            print(f"【{code}】已经结束")
        times += 1
        time.sleep(0.3)
        # 随机等待
        # time.sleep(random.randint(1, 10))


# 带异常过滤
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(300),
    retry=retry_if_exception_type(ConnectionError),
    reraise=True  # 重试后仍失败则抛出原始异常
)
def list_stock_codes():
    # 获取日线数据
    t = datetime.now().strftime("%Y-%m-%d") + " 00:00:00"
    t = '2025-06-16 00:00:00'
    sql = f"SELECT * FROM stock_basic_info where ts_code not in (select ts_code from stock_daily sbi where trade_date >= '{t}' group by ts_code)"
    print(sql)
    df = MysqlEngine.query_to_dataframe(sql)
    ts_codes = df['ts_code'].unique().tolist()
    print(ts_codes)
    return ts_codes


def main(start_date, end_date):
    # 增量更新股票
    try:
        print("开始采集股票基础信息")
        save_gp_basic()
        print("股票基础信息采集完成")
    except Exception as e:
        print(e)

    # 爬当天的日线数据
    start = start_date if start_date is not None else datetime.now().strftime("%Y%m%d")
    end = end_date if end_date is not None else datetime.now().strftime("%Y%m%d")
    print(f'采集日期：{start} - {end}')
    stock_codes = list_stock_codes()
    save_stock_daily(stock_codes, start_date=start, end_date=end)


if __name__ == '__main__':
    main(None  , None)
    print("采集完成")
    print(datetime.now().strftime("%Y%m%d"))
