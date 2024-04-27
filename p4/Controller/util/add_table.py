from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

topo = load_topo('topology.json')
switches = topo.get_p4switches()

def AddTableToAllSwitch(tablename,action,actionparam,matchkey):
    for sw in switches:
        thrift_port = topo.get_thrift_port(sw)
        switch_api = SimpleSwitchThriftAPI(thrift_port)
        switch_api.table_add(table_name=tablename,action_name=action,action_params=actionparam,match_keys=matchkey)

def DeleteTableToAllSwitch(tablename,matchkey):
    for sw in switches:
        thrift_port = topo.get_thrift_port(sw)
        switch_api = SimpleSwitchThriftAPI(thrift_port)
        switch_api.table_delete_match(table_name=tablename,match_keys=matchkey)