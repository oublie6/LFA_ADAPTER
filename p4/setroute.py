from collections import deque
from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

def main():
    # 初始化网络拓扑
    topo = load_topo('topology.json')
    switches = topo.get_p4switches()
    hosts = topo.get_hosts()

    # 初始化一个字典来存储每个交换机的API接口
    switch_api = {}
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


    # 对每个交换机运行BFS，并配置路由
    for host in hosts:
        print("当前配置路由节点",host)
        bfs(topo,host)