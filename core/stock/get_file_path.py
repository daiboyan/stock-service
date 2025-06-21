import os

if __name__ == '__main__':

    # 获取当前工作目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")

    # 在当前目录及其子目录中搜索文件
    target_file = 'current_stock_picks.csv'
    for root, dirs, files in os.walk(current_dir):
        if target_file in files:
            file_path = os.path.join(root, target_file)
            print(f"找到文件: {file_path}")

            # 读取文件
            with open(file_path, 'r') as file:
                content = file.read()
            print(content)
            break
    else:
        print(f"未找到文件: {target_file}")