import time
from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

topo = load_topo('topology.json')
switches = topo.get_p4switches()
tanzhenkaixiao=[]

totalbyte,totalpacket=0,0

def GetCounterInfo(sw):
    now=0
    totalbyte_before,totalpacket_before=0,0
    while True:
        print(now)
        now+=1
        totalbyte,totalpacket=0,0
        for sw in switches:
            thrift_port = topo.get_thrift_port(sw)
            switch_api = SimpleSwitchThriftAPI(thrift_port)
            res=switch_api.counter_read("prob_counter",0)
            totalbyte+=res[0]
            totalpacket+=res[1]
            print(sw,res)
        byteinc,packetinc=totalbyte-totalbyte_before,totalpacket-totalpacket_before
        totalbyte_before,totalpacket_before=totalbyte,totalpacket
        return(byteinc,packetinc)
    
while True:
    time.sleep(1)
    sum==0
    for sw in switches:
        byteinc,bytepacketinc=GetCounterInfo(sw)
        sum+=byteinc
    linklen=len(topo.edges)
    prob_cost=sum/linklen
    tanzhenkaixiao.append(prob_cost)
    