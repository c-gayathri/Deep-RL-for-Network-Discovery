import networkx as nx
import numpy as np
from icm import sample_live_icm, indicator, make_multilinear_objective_samples
from utils import greedy
from multiprocessing import Process, Manager, Pool

# icm from https://github.com/kage08/graph_sample_rl/blob/master/icm.py
# sample_live_icm(graph, n_samples) samples graphs with edges with individual probabilities graph[u][v]['p']
# make_multilinear_objective_samples returns sum( #target_nodes * probability_of_activation ) assuming all the nodes in a connected component are activated if
# ...atleast one node in the component is activated.
# indicator(S,n) returns a list of length n where the indices indicated in S are set to one and the others are zero

PROP_PROBAB = 0.1
BUDGET = 10
PROCESSORS = 8
SAMPLES = 100
# Above variables may be reset in train.py

def multi_to_set(f,g):
    '''
    Takes as input a function defined on indicator vectors of sets, and returns
    a version of the function which directly accepts sets
    '''
    def f_set(S):
        return f(indicator(S, len(g)))
    return f_set

def comobjfunction(graph, influencers, node_community, n_c, samples=1000):
        live_graphs = sample_live_icm(graph, samples) # create 1000 samples of graphs with edges with given probabilities
        comInfl = []
        # sum( #target_nodes * probability_of_activation ) assuming all the nodes in a cc are activated if atleast one node in the cc is activated.
        '''
        make_multilinear_objective_samples:

        Given a set of sampled live edge graphs, returns an function evaluating the 
        multilinear extension for the corresponding influence maximization problem.
        
        live_graphs: list of networkx graphs containing sampled live edges
        
        target_nodes: nodes that should be counted towards the objective
        
        selectable_nodes: nodes that are eligible to be chosen as seeds
        
        p_attend: probability that each node will be influenced if it is chosen as
        a seed.
        '''
        for c in range(n_c):
            comNodes = [i for i,com in enumerate(node_community) if com == c]
            f_multi = make_multilinear_objective_samples(live_graphs, comNodes, list(graph.nodes()), np.ones(len(graph)))
            f_set = multi_to_set(f_multi, graph)
            comInfl.append(f_set(influencers))
        return comInfl

def genoptfunction(graph, samples=1000):
        live_graphs = sample_live_icm(graph, samples) # create 1000 samples of graphs with edges with given probabilities
        # sum( #target_nodes * probability_of_activation ) assuming all the nodes in a cc are activated if atleast one node in the cc is activated.
        # print('Length of graph: ', len(graph))
        f_multi = make_multilinear_objective_samples(live_graphs, list(graph.nodes()), list(graph.nodes()), np.ones(len(graph)))
        f_set = multi_to_set(f_multi, graph)
        return f_set

def influence(graph, full_graph, samples=SAMPLES):
    for u,v in graph.edges():
        graph[u][v]['p']=PROP_PROBAB
    
    '''def genoptfunction(graph, samples=1000):
        live_graphs = sample_live_icm(graph, samples) # create 1000 samples of graphs with edges with given probabilities
        # sum( #target_nodes * probability_of_activation ) assuming all the nodes in a cc are activated if atleast one node in the cc is activated.
        f_multi = make_multilinear_objective_samples(live_graphs, list(graph.nodes()), list(graph.nodes()), np.ones(len(graph)))
        f_set = multi_to_set(f_multi, graph)
        return f_set'''
    
    f_set = genoptfunction(graph, samples)
    S, obj = greedy(list(range(len(graph))), BUDGET, f_set)
    # returns a 1) set of nodes chosen by greedily adding nodes to the set 
    # 2) achieved objective function - the number of target nodes * probability

    f_set1 = genoptfunction(full_graph, samples)
    opt_obj = f_set1(S) 

    return opt_obj, obj, S # returns 1) influence over full graph 2) over the current discovered graph & 3) the choice of greedily chosen influential nodes

# DEBUG
def influence_wrapper(l,g,fg,s,influence=influence):
    global ans # Debug
    # global l # Debug
    ans = influence(g,fg,s)
    l.append(ans[0])


'''def parallel_influence(graph, full_graph, times, samples=SAMPLES, influence=influence):

    def influence_wrapper(l,g,fg,s,influence=influence):
        global ans # Debug
        # global l # Debug
        ans = influence(g,fg,s)
        l.append(ans[0])
    
    l = Manager().list()
    # processes = [Process(target=influence_wrapper, args=(l, graph, full_graph, samples)) for _ in range(times)]

    #DEBUG
    #with Pool(processes = PROCESSORS) as pool:
    #processes = [Pool.map(influence_wrapper, (l, graph, full_graph, samples)) for _ in range(times)]
    processes = [Process(target=influence_wrapper, args=(l, graph, full_graph, samples)) for _ in range(times)]
    
    i=0
    while i<len(processes):
        j = i+PROCESSORS if i+PROCESSORS < len(processes) else len(processes)-1
        ps = processes[i:j]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        i+= PROCESSORS
    l = list(l)
    return np.mean(l)'''

def parallel_influence(graph, full_graph, times, samples=SAMPLES, influence=influence):

    '''def influence_wrapper(l,g,fg,s,influence=influence):
        global ans # Debug
        # global l # Debug
        ans = influence(g,fg,s)
        l.append(ans[0])'''
    
    l = Manager().list()
    # processes = [Process(target=influence_wrapper, args=(l, graph, full_graph, samples)) for _ in range(times)]

    #DEBUG
    #with Pool(processes = PROCESSORS) as pool:
    #processes = [Pool.map(influence_wrapper, (l, graph, full_graph, samples)) for _ in range(times)]
    processes = [Process(target=influence_wrapper, args=(l, graph, full_graph, samples)) for _ in range(times)]
    
    '''if __name__ == "expts.influence":
        i=0
        print('PARINF DEBUG: Entered main')
        while i<len(processes):
            j = i+PROCESSORS if i+PROCESSORS < len(processes) else len(processes)-1
            ps = processes[i:j]
            for p in ps:
                p.start()
            for p in ps:
                p.join()
            i+= PROCESSORS
        l = list(l)
        return np.mean(l)'''

    if __name__ == "expts.influence":
        i=0
        print('PARINF DEBUG: Entered main')
        while i<len(processes):
            j = i+PROCESSORS if i+PROCESSORS < len(processes) else len(processes)-1
            ps = processes[i:j]
            for p in ps:
                p.start()
            for p in ps:
                p.join()
            i+= PROCESSORS
        l = list(l)
        return np.mean(l)

        
if __name__ == "__main__":
    g = nx.erdos_renyi_graph(100,0.5)
    for u,v in g.edges():
        g[u][v]['p'] = 0.1
    import time
    
    '''print('--------------------- CODE PRINT ----------------------------\n\n\n\n\n')
    start = time.time()
    print(parallel_influence(g,g,10, 1000))
    end1 = time.time()
    print('Parallel took', end1-start, 'seconds')
    start = time.time()
    ls = [influence(g,g,100)[0] for _ in range(10)]
    print(np.mean(ls))
    end1 = time.time()
    print('Seq took', end1-start, 'seconds')
    print('\n\n\n\n\n--------------------- PRINT END ----------------------------')'''
