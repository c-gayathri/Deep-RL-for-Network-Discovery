import networkx as nx
import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle

ratios = [0.33,0.33,0.34]
g_size = 300
p_in = [0.06, 0.03, 0.03]
p_out = 0.005

communities = [0]*len(ratios)
node_community = []
sum = 0
for i,r in enumerate(ratios[:-1]):
    count = int(g_size*r)
    communities[i] = count
    sum += count
communities[-1] = g_size - sum


first = 0
second = communities[0]
g = nx.Graph()
for c,r in enumerate(ratios):
    for i in range(first, second):
        node_community.append(c)
        for j in range(g_size):
            if j in [first, second]:
                if random.random() < 0.7:
                    g.add_edge(i,j)
            else:
                if random.random() < 0.2:
                    g.add_edge(i,j)
    first += communities[c]
    if c+1 < len(ratios):
        second += communities[c+1]
    else:
        second = -1

# nx.draw(g)
# plt.savefig('sbm.png')

d = {'ratios': ratios, 'g_size': g_size, 'communities': communities, 'node_community': node_community, 'graph': g, 'p_in': p_in, 'p_out': p_out}
#db = pd.DataFrame(d)
db = pd.DataFrame({'ratios': ratios})

with open('graph_extreme.pickle', 'wb') as handle:
    pickle.dump(d, handle)

#nx.write_gpickle(g, 'sbm_400.gpickle')