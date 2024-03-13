import threading
from prob import Prob
from scapy.all import Ether, IP, TCP, sendp

# 定义一个函数用于从特定的网络接口发送数据包
def send_prob(iface,targetID,version=0):
    packet = Ether()/IP(dst="1.2.3.4")/Prob(targetID=targetID,util=0,version=version,transTime=0)
    sendp(packet, iface=iface)

def mutisend_prob(targetID,version):

    # 创建并启动两个线程，分别对应两个不同的网络接口
    thread1 = threading.Thread(target=send_prob, args=('eth0',))
    thread2 = threading.Thread(target=send_prob, args=('eth1',))
    thread1.start()
    thread2.start()

