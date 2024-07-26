import networkx as nx

def genNet(N=10,p=0.7):
# p: The probability of connecting any two nodes
    while True:
        Graph = nx.fast_gnp_random_graph(N, p)
        if nx.is_connected(Graph):  # is connected
            break
    nx.write_gml(Graph, "network.gml")


genNet()
