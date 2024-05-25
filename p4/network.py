import logging

from p4utils.mininetlib.network_API import NetworkAPI
from .topologyzoo.util import *

templateDir='p4/template/'
graphDir='p4/topologyzoo/topo/'



logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s')

import time
from pprint import pprint

iperfServerStr=r"bash -c 'iperf3 -s  -p 5201 &> iperfserver/"
iperfServerStrAtt=r"bash -c 'iperf3 -s -p 5202  &> iperfserver/"
iperfClientStr1=r"bash -c 'iperf3 -p 5201 -c  "
iperfClientStr2=r" -u -b "
iperfClientStr3=" -t   86400   -i 1 &> iperfclient/"
iperfClientStr4=" -t   30   -i 1 &> iperfclient/"

sendprobStr=r"bash -c 'python p4/util/sendpacket/loopsend.py &> sendprob/"

timeStart=time.perf_counter()
def GetNetworkBuildTime(graph):
        timeEnd=time.perf_counter()
        buildTime = (timeEnd - timeStart)*1000
        logging.info(f'{graph} 构建时间为: {buildTime}ms')
        # with open("app.log","a+") as f:
        #         f.write(graph+"构建时间为:"+str(buildTime)+"ms")

def BuildNet(graph,bw):

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
                        net.setBw(fromname,toname,bw)
                else:
                        print(fromname,toname,'出现重复链路')
                # for key, value in attrs.items():
                # print(f"    {key}: {value}")

        
        # 显示node之间的链路数
        for node1 in net.nodes():
                pprint(node1)
                for node2 in net.nodes():
                        if net.areNeighbors(node1, node2):
                                print(node1,node2,len(net._linkEntry(node1, node2)[0]))

        # net.addTask("s1",GetNetworkBuildTime,args={graph})
        hnow=0
        hostsss=net.hosts()
        t=len(hostsss)/4
        for h in hostsss:
                hnow+=1
                ipp="10."+str(hnow+1)+"."+str(hnow)+".2"
                net.addTask(name=h,exe=iperfServerStr+h+r"'",start=1)
                net.addTask(name=h,exe=iperfClientStr1+ipp+iperfClientStr2+" 100K "+iperfClientStr3+"normal/"+h+r"'",start=10)
                net.addTask(name=h,exe=iperfServerStrAtt+h+r"_att'",start=1)
                net.addTask(name=h,exe=iperfClientStr1+ipp + " -p 5202 "+iperfClientStr2+" 500K "+iperfClientStr4+"attack/"+h+r"_att'",start=30)
                net.addTask(name=h,exe=iperfClientStr1+ipp + " -p 5202 "+iperfClientStr2+" 500K "+iperfClientStr4+"attack/"+h+r"_att2'",start=90)
                # exit(0)
        
        # t=hnow/4
        # n=0
        # tmp=0
        # for h in net.hosts(sort=False):
        #         n+=1
        #         tmp=h[1:]
        #         tmp=int(tmp)
        #         ipp="10."+str(tmp+1)+"."+str(tmp)+".2"
        #         net.addTask(name=h,exe=iperfServerStrAtt+h+r"_att'",start=1)
        #         net.addTask(name=h,exe=iperfClientStr1+ipp + " -p 5202 "+iperfClientStr2+" 30M "+iperfClientStr4+"/attack/"+h+r"_att'",start=100)
                
        #         if n>=t:
        #                 break
        #         # exit(0)

        net.addTask(name="h13",exe=sendprobStr+"h13'",start=10)
        net.addTask(name="s1",exe=r'sudo  bash -c "python route.py &> route.log" &',start=1)
        net.addTask(name="s1",exe=r'sudo  bash -c "python add_server.py --targetIP 10.14.13.2/24 --targetID 3 &> add_server.log" &',start=2)
        net.addTask(name="s1",exe=r'sudo  bash -c "python start_rerouting.py --targetID 3 &> start_rerouting.log" &',start=50)

        # Assignment strategy
        net.l3()

        # Nodes general options
        net.enablePcapDumpAll()
        net.enableLogAll()

        # Start network
        net.startNetwork()

