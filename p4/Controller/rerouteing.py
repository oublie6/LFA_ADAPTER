from .util.register_operation import *


def StartRerouting(targetID):
    WriteRegister("lfa_on",targetID,1)

def EndRerouting(targetID):
    WriteRegister("lfa_on",targetID,0)

def Max(a,b):
    if a>b:
        return a
    return b

def SelectedRerouting(h):
    topo = load_topo('topology.json')
    switches = topo.get_p4switches()
    targetIP=topo.get_host_ip(h)
    switch_api = {}
    for sw in switches:
        thrift_port = topo.get_thrift_port(sw)
        switch_api[sw] = SimpleSwitchThriftAPI(thrift_port)
        for neighbor in topo.get_neighbors(sw):
            if not topo.isP4Switch(neighbor):
                continue
            dst_port=topo.node_to_node_port_num(sw,neighbor)
            linkUtil[sw][neighbor]=switch_api[sw].register_read(register_name="local_util",index=thrift_port,show=True)
    linkUtil=map()
    visited=map()
    bestUtil=map()
    bestNhop=map()
    bestNhopMAC=map()
    for k in linkUtil.keys():
        visited[k]=False
        bestUtil[k]=float("+inf")
    directSwitch=switches[0]
    for n in topo.get_neighbors(h):
        directSwitch=n
    bestUtil[directSwitch]=0
    visited[directSwitch]=True
    for neighbor in topo.get_neighbors(directSwitch):
        if not topo.isP4Switch(neighbor):
            continue
        if linkUtil[neighbor][directSwitch] < bestUtil[neighbor]:
            bestUtil[neighbor][directSwitch]=linkUtil[neighbor][directSwitch]
            bestNhop=topo.node_to_node_port_num(neighbor,directSwitch)
            bestNhopMAC=topo.node_to_node_mac(directSwitch,neighbor)
    for i in range(len(switches)-1):
        minUtil = float("+inf")
        minSw=directSwitch
        for sw in switches:
            if visited[sw]==True:
                continue
            if bestUtil[sw]<minUtil:
                minUtil=bestUtil[sw]
                minSw=sw
        visited[minSw]=True
        for neighbor in topo.get_neighbors(minSw):
            if not topo.isP4Switch(neighbor):
                continue
            if linkUtil[neighbor][minSw] < bestUtil[neighbor]:
                bestUtil[neighbor][minSw]=linkUtil[neighbor][minSw]
                bestNhop=topo.node_to_node_port_num(neighbor,minSw)
                bestNhopMAC=topo.node_to_node_mac(minSw,neighbor)
            switch_api[minSw].table_delete_match('ipv4_lpm', 'ipv4_forward', [targetIP])
            switch_api[minSw].table_add('ipv4_lpm', 'ipv4_forward', [targetIP], action_params=[bestNhopMAC[minSw],hex(bestNhop[minSw])])
                
    