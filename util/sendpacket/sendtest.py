import threading
from scapy.all import Ether, IP, TCP, sendp

# 定义一个函数用于从特定的网络接口发送数据包
def send_prob(iface,targetID,version=0):
    packet = Ether()/IP(dst="1.2.3.4")/Prob(targetID=targetID,util=0,version=version,transTime=0)
    sendp(packet, iface=iface)

def sendIPV4(dst):

