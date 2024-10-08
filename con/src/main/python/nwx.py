import networkx as nx
#import matplotlib.pyplot as plt
import csv

import sys
csv.field_size_limit(sys.maxsize)

lang = "fra"
split = "test"

U = nx.DiGraph()
nodes = set()
edges = []
path = f"../../../dat/dep/{lang}-{split}.tsv"
with open(path) as file:
  lines = csv.reader(file, delimiter='\t', quotechar="Đ")
  for line in lines:
    # print(line)
    nodes.add(line[0])
    nodes.add(line[1])
    edges.append((line[0], line[1], {"label": line[2]}))
    
U.add_nodes_from(nodes)
U.add_edges_from(edges)
print(f"Number of nodes in {split} split of {lang} = ", len(nodes))
print(f"Number of edges in {split} split of {lang} = ", len(edges))


# # pos = nx.circular_layout(U)
# pos = nx.spring_layout(U, seed=992015)
# plt.figure(1, figsize=(20,20))
# # nodes
# nx.draw_networkx_nodes(U, pos, node_size=600, node_color='blue', alpha=0)
# nx.draw_networkx_labels(U, pos, font_size=8, font_family="courier")
# # edges
# nx.draw_networkx_edges(U, pos, width=1, arrowstyle='->')
# edge_labels = nx.get_edge_attributes(U, "label")
# nx.draw_networkx_edge_labels(U, pos, edge_labels, font_size=5)

nx.is_directed_acyclic_graph(U)

# nx.simrank_similarity(U, 'outside:ADP', "near:ADP") # slow for the training graph

ic = nx.in_degree_centrality(U)

with open(f"../../../dat/dep/{lang}-indegree-centrality-{split}.tsv", 'w') as file: 
    w = csv.writer(file, delimiter='\t')
    w.writerows(ic.items())

ec = nx.eigenvector_centrality(U)

# write the values to a cvs file
with open(f"../../../dat/dep/{lang}-eigenvector-centrality-{split}.tsv", 'w') as file: 
    w = csv.writer(file, delimiter='\t')
    w.writerows(ec.items())

pr = nx.pagerank(U)

with open(f"../../../dat/dep/{lang}-pagerank-{split}.tsv", 'w') as file: 
    w = csv.writer(file, delimiter='\t')
    w.writerows(pr.items())

# pa = nx.preferential_attachment(U)




