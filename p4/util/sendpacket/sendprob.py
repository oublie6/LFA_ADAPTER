import threading

from prob import Prob
from scapy.all import *

# 获取本地网卡

interfaces = get_if_list()

selected_iface = None
for iface in interfaces:
    if "lo" not in iface:
        selected_iface = iface
        break

# 定义一个函数用于从特定的网络接口发送数据包
def send_prob(iface,targetID,version=0):
    packet = Ether()/IP(dst="1.2.3.4",proto=254)/Prob(targetID=targetID,util=0,version=version,transTime=0)
    sendp(packet, iface=iface)

def mutisend_prob(targetID,version):

    # 创建并启动两个线程，分别对应两个不同的网络接口
    thread1 = threading.Thread(target=send_prob, args=('eth0',))
    thread2 = threading.Thread(target=send_prob, args=('eth1',))
    thread1.start()
    thread2.start()
def sendone():
    # 选择第一个非回环网卡接口发送数据包
    if selected_iface:
        print(f"Selected interface: {selected_iface}")
        send_prob(selected_iface, targetID=1, version=100)
    else:
        print("No non-loopback interfaces found.")

import time


def sendloop(t,targetID):

    # 获取本地网卡

    interfaces = get_if_list()

    selected_iface = None
    for iface in interfaces:
        if "lo" not in iface:
            selected_iface = iface
            break
   
    # 选择第一个非回环网卡接口发送数据包
    if selected_iface:
        print(f"Selected interface: {selected_iface}")
        theversion=100
        while True:
            send_prob(selected_iface, targetID=targetID, version=theversion)
            theversion+=1
            time.sleep(t)
    else:
        print("No non-loopback interfaces found.")