import pandas as pd
from core.utils.db.mysql_engine import MysqlEngine
import os

def save_stock_list():
    # 读取 CSV 文件
    df = pd.read_csv("gp.csv", encoding="utf-8")  # 处理中文需指定编码
    print(df.head())  # 查看前几行数据

    # 将 DataFrame 写入数据库表（自动创建表或追加数据）
    df.to_sql(
        name="stock_basic_info",  # 表名
        con=MysqlEngine.create_engine(),
        if_exists="append",  # 表存在时追加数据；可选 "replace"（覆盖）或 "fail"（报错）
        index=False,         # 不写入 DataFrame 的索引
        chunksize=1000      # 分块写入提升性能
    )


if __name__ == '__main__':
    save_stock_list()

