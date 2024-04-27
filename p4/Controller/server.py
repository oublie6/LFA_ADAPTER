from .util.add_table import *


def add_server(targetIP,targetID):
    AddTableToAllSwitch(tablename="lfa_server",action="get_lfa_server",actionparam=[targetID],matchkey=[targetIP])

def delete_server(targetIP):
    DeleteTableToAllSwitch(tablename="lfa_server",matchkey=[targetIP])