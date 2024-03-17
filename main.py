import argparse

from p4.network import *

# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description='这是一个演示程序')

# 添加命令行参数
parser.add_argument('--topo', type=str, help='topo名')

# 解析命令行参数
args = parser.parse_args()


BuildNet(graphDir+args.topo)