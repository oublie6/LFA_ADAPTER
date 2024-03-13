import threading
from collections import deque

from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

from .util.digest_controller import run_digest_controller

topo = load_topo('topology.json')
switches = topo.get_p4switches()
hosts = topo.get_hosts()

# 初始化一个字典来存储每个交换机的API接口
switch_api = {}
digest_api={}
for sw in switches:
    thrift_port = topo.get_thrift_port(sw)
    switch_api[sw] = SimpleSwitchThriftAPI(thrift_port)

threads = []

def main():
    for sw in switches:
        thread = threading.Thread(target=run_digest_controller, args=(sw,))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()
