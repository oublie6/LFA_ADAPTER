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
iperfClientStr1=r"bash -c 'iperf3   -c  "
iperfClientStr2=r" -u -b "
iperfClientStr3=r"  -i 1 -t  "
iperfClientStr4=r"&> iperfclient/"


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
                        net.setBw(fromname,toname,5)
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
        # for h in hostsss:
        #         if hnow ==13 or hnow ==15:
        #                 ipp="10."+str(hnow+1)+"."+str(hnow)+".2"
        #                 ips="10."+str(hnow+2)+"."+str(hnow+1)+".2"
        #                 net.addTask(name=,exe=iperfServerStr+h+r"'",start=1)
        #                 net.addTask(name=h,exe=iperfClientStr1+ipp+ "  -p 5201   "+iperfClientStr2+" 1M "+iperfClientStr3+" 100 "+iperfClientStr4+"normal/"+h+r"'",start=10)
        #                 net.addTask(name=h,exe=iperfServerStrAtt+h+r"_att'",start=1)
        #                 net.addTask(name=h,exe=iperfClientStr1+ipp + "  -p 5202 "+iperfClientStr2+" 10M "+iperfClientStr3+" 30 "+iperfClientStr4+"attack/"+h+r"_att'",start=20)
        #                 net.addTask(name=h,exe=iperfClientStr1+ipp + "  -p 5202 "+iperfClientStr2+" 10M "+iperfClientStr3+ " 30 " +iperfClientStr4+"attack/"+h+r"_att2'",start=60)
        #         hnow+=1
        # exit(0)
        h13=" 10.14.13.2 "
        h15=" 10.16.15.2 "
        h20="10.21.20.2"
        net.addTask(name="h13",exe=iperfServerStr+"shishichongluyou"+r"'",start=1)
        net.addTask(name="h17",exe=iperfClientStr1+h13+ "  -p 5201   "+iperfClientStr2+" 1M "+iperfClientStr3+" 200 "+iperfClientStr4+"normal/"+"shishichongluyou"+r"'",start=10)
        net.addTask(name="h5",exe=iperfServerStrAtt+"shishichongluyou"+r"_att'",start=1)
        net.addTask(name="h17",exe=iperfClientStr1+"10.6.5.2" + "  -p 5202 "+iperfClientStr2+" 20M "+iperfClientStr3+" 50 "+iperfClientStr4+"attack/"+"shishichongluyou"+r"_att'",start=20)
        net.addTask(name="h17",exe=iperfClientStr1+"10.6.5.2" + "  -p 5202 "+iperfClientStr2+" 20M "+iperfClientStr3+ " 50 " +iperfClientStr4+"attack/"+"shishichongluyou"+r"_att2'",start=75)
        net.addTask(name="h17",exe=iperfClientStr1+"10.6.5.2" + "  -p 5202 "+iperfClientStr2+" 20M "+iperfClientStr3+ " 50 " +iperfClientStr4+"attack/"+"shishichongluyou"+r"_att3'",start=130)

        net.addTask(name="h15",exe=iperfServerStr+"wufangyu"+r"'",start=1)
        net.addTask(name="h18",exe=iperfClientStr1+h15+ "  -p 5201   "+iperfClientStr2+" 1M "+iperfClientStr3+" 200 "+iperfClientStr4+"normal/"+"wufangyu"+r"'",start=10)
        net.addTask(name="h15",exe=iperfServerStrAtt+"wufangyu"+r"_att'",start=1)
        net.addTask(name="h18",exe=iperfClientStr1+h15 + "  -p 5202 "+iperfClientStr2+" 20M "+iperfClientStr3+" 50 "+iperfClientStr4+"attack/"+"wufangyu"+r"_att'",start=20)
        net.addTask(name="h18",exe=iperfClientStr1+h15 + "  -p 5202 "+iperfClientStr2+" 20M "+iperfClientStr3+ " 50 " +iperfClientStr4+"attack/"+"wufangyu"+r"_att2'",start=75)
        net.addTask(name="h18",exe=iperfClientStr1+h15 + "  -p 5202 "+iperfClientStr2+" 20M "+iperfClientStr3+ " 50 " +iperfClientStr4+"attack/"+"wufangyu"+r"_att3'",start=130)

        net.addTask(name="h20",exe=iperfServerStr+"ripple"+r"'",start=1)
        net.addTask(name="h3",exe=iperfClientStr1+h20+ "  -p 5201   "+iperfClientStr2+" 1M "+iperfClientStr3+" 200 "+iperfClientStr4+"normal/"+"ripple"+r"'",start=10)
        net.addTask(name="h19",exe=iperfServerStrAtt+"ripple"+r"_att'",start=1)
        net.addTask(name="h3",exe=iperfClientStr1+"10.20.19.2" + "  -p 5202 "+iperfClientStr2+" 20M "+iperfClientStr3+" 50 "+iperfClientStr4+"attack/"+"ripple"+r"_att'",start=20)
        net.addTask(name="h3",exe=iperfClientStr1+"10.20.19.2" + "  -p 5202 "+iperfClientStr2+" 20M "+iperfClientStr3+ " 50 " +iperfClientStr4+"attack/"+"ripple"+r"_att2'",start=75)
        net.addTask(name="h3",exe=iperfClientStr1+"10.20.19.2" + "  -p 5202 "+iperfClientStr2+" 20M "+iperfClientStr3+ " 50 " +iperfClientStr4+"attack/"+"ripple"+r"_att3'",start=130)

        # net.addTask(name="h1",exe="bash -c 'iperf3 -s &> iperfserver/h1'",start=1)
        # net.addTask(name="h2",exe="bash -c 'iperf3 -c 10.2.1.2 -u -b 4M &> iperfserver/h2'",start=20)
        
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

        net.addTask(name="s1",exe=r'sudo  bash -c "python route.py &> route.log" &',start=1)

        net.addTask(name="h13",exe=r"bash -c 'python p4/util/sendpacket/loopsend.py --targetID 3 &> sendprob/"+"h13'",start=10)
        net.addTask(name="s1",exe=r'sudo  bash -c "python add_server.py --targetIP 10.14.13.2/24 --targetID 3 &> add_server.log" &',start=3)
        net.addTask(name="s1",exe=r'sudo  bash -c "python start_rerouting.py --targetID 3 &> start_rerouting.log" &',start=30)

        net.addTask(name="h20",exe=r"bash -c 'python p4/util/sendpacket/loopsend.py --targetID 2 &> sendprob/"+"h20'",start=10)
        net.addTask(name="s1",exe=r'sudo  bash -c "python add_server.py --targetIP 10.21.20.2/24 --targetID 2 &> add_server.log" &',start=3)
        net.addTask(name="s1",exe=r'sudo  bash -c "python start_rerouting.py --targetID 2 &> start_rerouting.log" &',start=30)

        # Assignment strategy
        net.l3()

        # Nodes general options
        net.enablePcapDumpAll()
        net.enableLogAll()

        # Start network
        net.startNetwork()

