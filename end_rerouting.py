import argparse

from p4.Controller.rerouteing import EndRerouting

# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description='关闭重路由')

# 添加命令行参数
parser.add_argument('--targetID', type=str, help='id')

# 解析命令行参数
args = parser.parse_args()

EndRerouting(args.targetID)