import time
from functools import wraps


def retry(max_attempts=3, delay=1, exceptions=(Exception,)):
    """
    重试装饰器

    参数:
        max_attempts: 最大尝试次数
        delay: 重试间隔（秒）
        exceptions: 触发重试的异常类型
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise  # 重试次数耗尽，抛出异常
                    print(f"尝试 {attempts}/{max_attempts} 失败，{delay}秒后重试... 错误: {e}")
                    time.sleep(delay)
            return None

        return wrapper

    return decorator


# 使用示例
@retry(max_attempts=5, delay=2, exceptions=(ConnectionError, TimeoutError))
def connect_to_service():
    # 模拟可能失败的操作
    import random
    if random.random() < 0.7:  # 70% 失败率
        raise ConnectionError("连接失败")
    return "连接成功"


print(connect_to_service())
