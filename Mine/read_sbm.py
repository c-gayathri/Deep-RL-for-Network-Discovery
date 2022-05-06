import networkx as nx
read = nx.read_gpickle("sbm.gpickle")

print(read.edges())