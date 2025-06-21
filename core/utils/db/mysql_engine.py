from pymysql.err import OperationalError, ProgrammingError
from sqlalchemy import create_engine, text
import pandas as pd
from urllib.parse import quote_plus


def optimize_dtypes(df):
    """优化DataFrame数据类型减少内存占用"""
    # 转换datetime类型
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # 转换数值类型为最小可存储类型
    for col in df.select_dtypes(include=['integer']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    return df


class MysqlEngine:
    @staticmethod
    def create_engine():
        # 配置数据库连接（以 MySQL 为例）
        # db_config = {
        #     "dialect": "mysql",
        #     "driver": "pymysql",
        #     "username": "root",
        #     "password": "Dai816846",
        #     "host": "localhost",
        #     "port": "3306",
        #     "database": "stock",
        # }

        db_config = {
            "dialect": "mysql",
            "driver": "pymysql",
            "username": "remoter",
            "password": quote_plus("Rem@9527!"),
            "host": "daiboyan.com",
            "port": "3306",
            "database": "stock",
        }

        # 创建连接引擎
        return create_engine(
            f"{db_config['dialect']}+{db_config['driver']}://"
            f"{db_config['username']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/"
            f"{db_config['database']}?charset=utf8mb4"
        )

    @staticmethod
    def query_to_dataframe(sql_query, params=None):
        """
        执行SQL查询并返回DataFrame

        参数：
        - sql_query: SQL查询语句（推荐使用参数化查询）
        - params: 查询参数（字典格式）

        返回：
        - pd.DataFrame: 查询结果
        - 或 None（查询失败时）
        """

        # 创建带连接池的引擎
        engine = MysqlEngine.create_engine()

        try:
            # 使用上下文管理器自动管理连接
            with engine.connect() as conn:
                # 使用text()实现参数绑定
                query = text(sql_query)

                # 执行查询（参数化防止SQL注入）
                df = pd.read_sql_query(
                    sql=query,
                    con=conn,
                    params=params,
                    coerce_float=True  # 自动转换数值类型
                )

                # 类型优化（可选）
                df = optimize_dtypes(df)

                return df

        except OperationalError as e:
            print(f"连接失败: {str(e)}")
        except ProgrammingError as e:
            print(f"SQL语法错误: {str(e)}")
        except Exception as e:
            print(f"未知错误: {str(e)}")
        finally:
            # 显式关闭引擎（长时间运行的程序需要）
            if 'engine' in locals():
                engine.dispose()

        return None


if __name__ == '__main__':
    en = MysqlEngine.create_engine()
    # en.connect()
    # en.dispose()
    # 示例1：基础查询
    simple_df = MysqlEngine.query_to_dataframe("SELECT * FROM stock_basic_info LIMIT 10")
    stock_codes = simple_df['ts_code'].unique().tolist()
    print(simple_df)

    # # 示例2：参数化查询（防止SQL注入）
    # params = {
    #     "start_date": "2023-01-01",
    #     "end_date": "2023-12-31",
    #     "min_amount": 1000
    # }
    #
    # complex_df = MysqlEngine.query_to_dataframe("""
    #                                 SELECT *
    #                                 FROM transactions
    #                                 WHERE transaction_date BETWEEN :start_date AND :end_date
    #                                   AND amount > :min_amount
    #                                 """, params=params)