from util.register_operation import *


def StartAttackSource(sw,port):
    thrift_port = topo.get_thrift_port(sw)
    switch_api = SimpleSwitchThriftAPI(thrift_port)
    switch_api.register_write("attack_source_mode",port,1)

def EndAttackSource(sw,port):
    thrift_port = topo.get_thrift_port(sw)
    switch_api = SimpleSwitchThriftAPI(thrift_port)
    switch_api.register_write("attack_source_mode",port,0)

