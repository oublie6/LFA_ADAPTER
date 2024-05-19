from .util.register_operation import *


def StartRerouting(targetID):
    WriteRegister("lfa_on",targetID,1)

def EndRerouting(targetID):
    WriteRegister("lfa_on",targetID,0)
