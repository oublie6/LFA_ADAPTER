from run_digest_all import *
from util.register_operation import *


def StartAttackSource(sw,port):
    thrift_port = topo.get_thrift_port(sw)
    switch_api = SimpleSwitchThriftAPI(thrift_port)
    switch_api.register_write("attack_source_mode",port,1)

def EndAttackSource(sw,port):
    thrift_port = topo.get_thrift_port(sw)
    switch_api = SimpleSwitchThriftAPI(thrift_port)
    switch_api.register_write("attack_source_mode",port,0)

def DetectAttackSource(suspiciousSet):
    maliciousIP=[]
    countMean=countTraceroute.getMean()
    pairMean=pairTraceroute.getMean()
    for ip in suspiciousSet:
        count=countTraceroute[ip]
        if count>countMean:
            pair=pairTraceroute[ip]
            if pair>pairMean:
                detect=detectIP[ip]+1
                detectIP[ip]=detect
                if detect>1:
                    maliciousIP.append(ip)
                    maliciousScore[ip]+=1
    return maliciousIP
