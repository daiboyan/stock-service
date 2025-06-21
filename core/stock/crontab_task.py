from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime

def daily_job():
    print(f"每日任务执行时间: {datetime.now()}")

# 创建调度器
scheduler = BlockingScheduler()

# 添加每日下午5点执行的任务
scheduler.add_job(
    daily_job,
    'cron',  # 使用cron表达式
    hour=17,  # 17点（24小时制）
    minute=0,  # 0分
    second=0   # 0秒
)

print("调度器已启动，每天下午5点执行任务...")
try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    print("程序已退出")