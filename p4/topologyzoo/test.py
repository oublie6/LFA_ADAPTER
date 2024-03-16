import networkx as nx

from util import ParseGraph, PrintGraphAttributes

# from util import draw_nx

# G = nx.read_graphml("topo/condensed_west_europe_delete.graphml")

# nolongitude, nolaitude = draw_nx(G, 10, 40, dlon=2, dlat=50, prefix="S")
# print("nolongitude", nolongitude)
# print("nolaitude", nolaitude)


G=ParseGraph("topo/AttMpls.graphml")
# print(G.nodes)
PrintGraphAttributes(G)