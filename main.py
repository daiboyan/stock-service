from fastapi import FastAPI
from core.stock.save_stock_info import main as stock_main

# 创建FastAPI应用实例
app = FastAPI()


# 定义一个根路径的GET接口
@app.get("/")
async def read_root():
    return {"message": "Hello, World"}


@app.get("/save")
async def read_items(start_date: str, end_date: str):
    stock_main(start_date=start_date, end_date=end_date)
