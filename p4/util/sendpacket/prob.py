from scapy.all import *


class Prob(Packet):
    name = "Prob"
    fields_desc = [
        IntField("targetID", 0),
        IntField("util", 0),
        IntField("version", 0),
        IntField("transTime", 0)
    ]