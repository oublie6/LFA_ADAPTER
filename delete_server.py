import argparse

from p4.Controller.server import delete_server

# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description='删除服务器')

# 添加命令行参数
parser.add_argument('--targetIP', type=str, help='ip')

# 解析命令行参数
args = parser.parse_args()

delete_server(args.targetIP)