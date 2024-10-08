import networkx as nx
import itertools
from numpy import array
import copy
#x1 x2 x3 x1x2
def pde_strategy(variables:list, input_dim:int, dim:int):
    numbers = [i+1 for i in range(input_dim)]
    # print(numbers)
    combinations_list = []

    for i in range(dim+1):
        combinations_list += list(itertools.combinations_with_replacement(numbers,i))
    u = [''.join(['x'+ str(combinations_list[i][j]) for j in range(len(combinations_list[i]))]) for i in range(len(combinations_list))]


    def adjust_weights(G, flow_dict, base_weight):
        """ Adjust weights based on current flow. """
        for u, v in G.edges():
            current_flow = flow_dict[u][v]
            # Adjust the weight: if there is flow, divide the base weight by the flow
            G[u][v]['weight'] = base_weight[(u, v)] / current_flow * 1000 if current_flow != 0 else base_weight[(u, v)] * 1000

    def calculate_total_cost(G, flow_dict):
        """ Calculate the total cost of the current flow. """
        total_cost = 0
        for u in flow_dict:
            for v in flow_dict[u]:
                if flow_dict[u][v] > 0:
                    total_cost += G[u][v]['weight']
        return total_cost

    def min_cost_flow_with_adjustment(G, source, sink, max_iter=20, tolerance=1e-4):
        base_weight = {edge: G[edge[0]][edge[1]]['weight'] for edge in G.edges()}
        previous_total_cost = float('inf')

        for i in range(max_iter):  # Iterate up to max_iter times
            # Compute the minimum cost flow
            flow_dict = nx.min_cost_flow(G)

            # Calculate the current total cost
            current_total_cost = calculate_total_cost(G, flow_dict)
            # Check for convergence
            if abs(previous_total_cost - current_total_cost) < tolerance:
                break
            # Adjust the weights based on the flow
            adjust_weights(G, flow_dict, base_weight)

            # Update the previous total cost
            previous_total_cost = current_total_cost

        return flow_dict

    G = nx.DiGraph()

    for i in range(dim):
        lower_dim = [x  for x in combinations_list if len(x) == i]
        higher_dim = [x  for x in combinations_list if len(x) == i+1]
        for j in range(len(higher_dim)):
            for k in range(len(lower_dim)):
                list1 = copy.deepcopy(list(lower_dim[k]))
                list2 = copy.deepcopy(list(higher_dim[j]))
                for l in range(len(list1)):
                    if list1[l] in list2:
                        list2.remove(list1[l])
                if len(list2) == 1:
                    # print('s' if i == 0 else 'u'+ ''.join(['x'+str(lower_dim[k][l]) for l in range(len(lower_dim[k]))]), 'u'+''.join(['x'+str(higher_dim[j][l]) for l in range(len(higher_dim[j]))]))
                    G.add_edge('s' if i == 0 else 'u'+''.join(['x'+str(lower_dim[k][l]) for l in range(len(lower_dim[k]))]), 'u'+''.join(['x'+str(higher_dim[j][l]) for l in range(len(higher_dim[j]))]), capacity=1000, weight=1000)

    #add important edges
    for var in variables:
        G.add_edge(var,'t',capacity=1, weight=0)

    # Add node demands (negative values for supply, positive for demand)
    G.nodes['t']['demand'] = len(variables)
    G.nodes['s']['demand'] = -1*len(variables)
    flow_dict = min_cost_flow_with_adjustment(G, 's', 't')
    uls = []
    m =[]
    d = {}
    for u, v, data in G.edges(data=True):
        if flow_dict[u][v] !=0:
            uls.append(u.replace('s','u'))
            if v != 't':
                str1 = copy.copy(v)
                #CHANGE: EVERY 2
                i=0
                while i <len(u):
                    if u[i] == 's' or u[i] == 'u':
                        rep = 'u'
                    else:
                        rep = u[i:i+2]
                    str1 = str1.replace(rep, '', 1)
                    i+= len(rep)
                # print((u.replace('s','u'),str1,v))
                m.append((u.replace('s','u'),str1,v))
                temp = u.replace('s','u')
                if not temp in d.keys():
                    d[temp] = [str1]
                else:  d[temp].append(str1)
    print(d)
    x = list(set(uls))
    return x, d
