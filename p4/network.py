from p4utils.mininetlib.network_API import NetworkAPI
from p4.topologyzoo.util import *

templateDir='p4/template/'
graphDir='p4/topologyzoo/topo/'

# def Main():

#         net = NetworkAPI()

#         # Network general options
#         net.setLogLevel('info')
#         net.enableCli()

#         # Network definition
#         net.addP4Switch('s1', cli_input=templateDir+'s1-commands.txt')
#         net.addP4Switch('s2', cli_input=templateDir+'s2-commands.txt')
#         net.addP4Switch('s3', cli_input=templateDir+'s3-commands.txt')

#         import os
#         print("当前工作目录：", os.getcwd())

#         net.setP4SourceAll(templateDir+'switch.p4')

#         net.addHost('h1')
#         net.addHost('h2')
#         net.addHost('h3')
#         net.addHost('h4')

#         net.addLink('h1','s1')
#         net.addLink('h2','s2')
#         net.addLink('s1','s2')
#         net.addLink('s1','s3')
#         net.addLink('h3','s3')
#         net.addLink('h4','s3')

#         # Assignment strategy
#         net.l3()

#         # Nodes general options
#         net.enablePcapDumpAll()
#         net.enableLogAll()

#         # Start network
#         net.startNetwork()

def BuildNet(graph):

        G=ParseGraph(graph)

        net = NetworkAPI()

        # Network general options
        net.setLogLevel('info')
        net.enableCli()
        
        # 添加交换机节点和主机
        print("添加交换机节点和主机:")
        for node, attrs in G.nodes(data=True):
                Sname='s'+node
                Hname='h'+node
                print(Sname,Hname)
                print("Node:", Sname)
                net.addP4Switch(Sname,cli_input=templateDir+'s-command.txt')
                net.addHost(Hname)
                net.addLink(Hname,Sname)
                # for key, value in attrs.items():
                # print(f"    {key}: {value}")

        net.setP4SourceAll(templateDir+'switch.p4',)

        # 添加链路
        mygraph=dict()

        print("\n添加链路:")
        for u, v, attrs in G.edges(data=True):
                fromname='s'+u
                toname='s'+v
                if fromname not in mygraph:
                        mygraph[fromname]=set()
                if toname not in mygraph:
                        mygraph[toname]=set()
                if toname not in mygraph[fromname]:
                        mygraph[fromname].add(toname)
                        mygraph[toname].add(fromname)
                        print(fromname,toname)
                        net.addLink(fromname,toname)
                else:
                        print(fromname,toname,'出现重复链路')
                # for key, value in attrs.items():
                # print(f"    {key}: {value}")
        
        # 设置链路带宽，单位为Mbps
        net.setBwAll(5)

        from pprint import pprint

        # 显示node之间的链路数
        for node1 in net.nodes():
                pprint(node1)
                for node2 in net.nodes():
                        if net.areNeighbors(node1, node2):
                                print(node1,node2,len(net._linkEntry(node1, node2)[0]))

        # Assignment strategy
        net.l3()

        # Nodes general options
        net.enablePcapDumpAll()
        net.enableLogAll()

        # Start network
        net.startNetwork()