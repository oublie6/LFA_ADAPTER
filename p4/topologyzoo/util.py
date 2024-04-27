import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.basemap import Basemap as Basemap


# 解析topo图
def ParseGraph(filepath):
    graph=nx.read_graphml(filepath)
    print(filepath,"节点数:",len(graph.nodes()))
    print(filepath,"边数:",len(graph.edges()))
    return  graph

# 打印节点属性和边属性
def PrintGraphAttributes(graph):
    # 打印节点属性
    print("Node attributes:")
    for node, attrs in graph.nodes(data=True):
        print("Node:", node)
        for key, value in attrs.items():
            print(f"    {key}: {value}")

    # 打印边属性
    print("\nEdge attributes:")
    for u, v, attrs in graph.edges(data=True):
        print(f"Edge ({u}, {v})")
        for key, value in attrs.items():
            print(f"    {key}: {value}")

def draw_one(graph="AttMpls.graphml"):
    G = nx.read_graphml("./topo/"+graph)

    pos = {}

    for node in G.nodes():
        longitude = G.nodes[node].get('Longitude')
        if longitude == None:
            longitude = 1
        latitude = G.nodes[node].get('Latitude')
        if latitude == None:
            latitude = 1
        
        pos[node]=[longitude,latitude]


    nx.draw_networkx(G, pos, node_size=100, node_color='#1E90FF', font_size=10)

def draw_one2(graph="AttMpls.graphml"):
    G = nx.read_graphml("./topo/"+graph)

    pos = {}
    nodeColors=[]
    sum=len(G.nodes())
    bots=int(sum/4)
    decoy=int(sum/4)
    rest=sum-bots-decoy-1
    nodeColors=["green"]
    for i in range(bots):
        nodeColors.append("red")
    for i in range(decoy):
        nodeColors.append("yellow")
    for i in range(rest):
        nodeColors.append('#1E90FF')
    random.shuffle(nodeColors)
    colerToNode=defaultdict(list)
    for i in range(sum):
        colerToNode[nodeColors[i]].append(i)
    for k in colerToNode:
        print(k,colerToNode[k])

    for node in G.nodes():
        longitude = G.nodes[node].get('Longitude')
        if longitude == None:
            longitude = 1
        latitude = G.nodes[node].get('Latitude')
        if latitude == None:
            latitude = 1
        
        pos[node]=[longitude,latitude]


    nx.draw_networkx(G, pos, node_size=100, node_color=nodeColors, font_size=10)

    plt.show()

def draw_nx(G, llon=10, llat=40, ulon=30, ulat=55, dlon=1, dlat=1, ns=10, nc='red', fs=10, prefix=None ):
    m = Basemap(
        projection='merc',
        llcrnrlon=-llon,
        llcrnrlat=llat,
        urcrnrlon=ulon,
        urcrnrlat=ulat,
        lat_ts=0,
        resolution='i',
        suppress_ticks=True)

    lats = []
    lons = []
    nolongitude = []
    nolatitude = []

    for node in G.nodes():
        longitude = G.nodes[node].get('Longitude')

        if longitude == None:
            nolongitude.append(node)
            longitude = dlon
        latitude = G.nodes[node].get('Latitude')
        if latitude == None:
            nolatitude.append(node)
            latitude = dlat
        lats.append(latitude)
        lons.append(longitude)

    mx, my = m(lons, lats)

    pos = {}
    i = 0

    for node in G.nodes():
        pos[node] = [mx[i], my[i]]
        i += 1

    nx.draw_networkx(G, pos, node_size=ns, node_color=nc, font_size=fs,
    labels={n:prefix+n for n in G})

    m.drawcountries()
    m.drawstates()
    m.bluemarble()

    plt.show()

    return nolongitude,nolatitude