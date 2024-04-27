import argparse

from p4.Controller.server import add_server

# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description='添加服务器')

# 添加命令行参数
parser.add_argument('--targetIP', type=str, help='ip')
parser.add_argument('--targetID', type=str, help='id')

# 解析命令行参数
args = parser.parse_args()

add_server(args.targetIP,args.targetID)