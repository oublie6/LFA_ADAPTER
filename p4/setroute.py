from collections import deque

from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

switch_api = {}

def main():
    
    # 初始化网络拓扑
    topo = load_topo('topology.json')
    switches = topo.get_p4switches()
    hosts = topo.get_hosts()

    # 初始化一个字典来存储每个交换机的API接口
    
    for sw in switches:
        thrift_port = topo.get_thrift_port(sw)
        switch_api[sw] = SimpleSwitchThriftAPI(thrift_port)

    # 广度优先搜索函数
    def bfs(topo, start):
        visited = {start: None}
        queue = deque([start])
        ip_start = topo.get_host_ip(start)+'/24'

        while queue:
            print(queue)
            vertex = queue.popleft()
            for neighbor in topo.get_neighbors(vertex):
                if not topo.isP4Switch(neighbor):
                    continue
                if neighbor not in visited:
                    visited[neighbor] = vertex
                    dst_mac=topo.node_to_node_mac(vertex,neighbor)
                    dst_port=topo.node_to_node_port_num(neighbor,vertex)
                    print(ip_start,dst_mac,dst_port)
                    switch_api[neighbor].table_add('ipv4_lpm', 'ipv4_forward', [ip_start], action_params=[dst_mac,hex(dst_port)])
                    queue.append(neighbor)
        return visited


    # 对每个目标主机运行BFS，并配置路由
    for host in hosts:
        print("当前配置路由节点",host)
        bfs(topo,host)

    # 对每个交换机配置广播
    for sw in switches:
        port=set()
        hostsports=set()
        for neighbor in topo.get_neighbors(sw):
            if topo.isP4Switch(neighbor):
                port.add(topo.node_to_node_port_num(sw,neighbor))
            else:
                hostsports.add(topo.node_to_node_port_num(sw,neighbor))
        if len(port)>1:
            print("sw,port",sw,port)
            for p in port:
                print(p)
                mcastports=list(port-{p})
                switch_api[sw].mc_mgrp_create(p)
                node_handle=switch_api[sw].mc_node_create(p,mcastports)
                switch_api[sw].mc_node_associate(p,node_handle)
                switch_api[sw].table_add("hulapp_mcast","set_hulapp_mcast",{hex(p)},{hex(p)})
        if len(hostsports)>0:
            print("sw,hostsports",sw,hostsports)
            for p in hostsports:
                print(p)
                mcastports=port
                switch_api[sw].mc_mgrp_create(p)
                node_handle=switch_api[sw].mc_node_create(p,mcastports)
                switch_api[sw].mc_node_associate(p,node_handle)
                switch_api[sw].table_add("hulapp_mcast","set_hulapp_mcast",{hex(p)},{hex(p)})

            

        
        
