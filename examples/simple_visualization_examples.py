import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from hiwenet import hiwenet

dimensionality = 1000
num_groups = 5
num_links = int(num_groups*(num_groups-1)/2.0)

random_indices_into_groups = np.random.randint(0, num_groups, [1, dimensionality])
group_ids = np.arange(num_groups)
groups = group_ids[random_indices_into_groups].flatten()

features = 1000*np.random.random(dimensionality)

G = hiwenet(features, groups, weight_method = 'histogram_intersection',
              return_networkx_graph=True)

edge_weights = np.array([d['weight'] for (u,v,d) in G.edges(data=True) ])

elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >  np.mean(edge_weights)]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <= np.mean(edge_weights)]

pos=nx.spring_layout(G) # positions for all nodes

plt.figure()
# nodes
nx.draw_networkx_nodes(G,pos,node_size=700)

# edges
nx.draw_networkx_edges(G,pos,edgelist=elarge, width=6)
nx.draw_networkx_edges(G,pos,edgelist=esmall, width=6, alpha=0.5,edge_color='b',style='dashed')

# labels
nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')
plt.show(block=False)

## edge weight colormap
plt.figure()
pos=nx.spring_layout(G)
colors=range(num_links)
nx.draw(G,pos,node_color='#A0CBE2',edge_color=colors,width=4,edge_cmap=plt.cm.Blues,with_labels=False)
plt.show(block=False)

## degree rank plot
plt.figure()
degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
#print "Degree sequence", degree_sequence
dmax=max(degree_sequence)
plt.loglog(degree_sequence,'b-',marker='o')
plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")
# draw graph in inset
plt.axes([0.45,0.45,0.45,0.45])
Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
pos=nx.spring_layout(Gcc)
plt.axis('off')
nx.draw_networkx_nodes(Gcc,pos,node_size=20)
nx.draw_networkx_edges(Gcc,pos,alpha=0.4)
nx.draw_networkx_nodes(Gcc,pos,node_size=20)
nx.draw_networkx_edges(Gcc,pos,alpha=0.4)
plt.show(block=False)