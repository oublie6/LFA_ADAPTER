from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

topo = load_topo('topology.json')
switches = topo.get_p4switches()

def ResetRegister(registername):
    for sw in switches:
        thrift_port = topo.get_thrift_port(sw)
        switch_api = SimpleSwitchThriftAPI(thrift_port)
        switch_api.register_reset(register_name=registername)

def WriteRegister(registername,theindex,thevalue):
    for sw in switches:
        thrift_port = topo.get_thrift_port(sw)
        switch_api = SimpleSwitchThriftAPI(thrift_port)
        switch_api.register_write(register_name=registername,index=theindex,value=thevalue)

def ReadRegister(registername,theindex):
    ret=map()
    for sw in switches:
        thrift_port = topo.get_thrift_port(sw)
        switch_api = SimpleSwitchThriftAPI(thrift_port)
        ret[sw]=switch_api.register_read(register_name=registername,index=theindex,show=True)
    return ret