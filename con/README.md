# Steps

1. Run `vlp.dep.NetworkX` to read a UD treebank and export its graph edges to a TSV file for a language:
   
   `bloop run -p con -m vlp.dep.NetworkX`
   
  The statistic of nodes and edges for 3 treebanks is as follows:

  | Language  | train. nodes | train. edges | dev. nodes | dev. edges | test nodes | test edges |
  |-------|-------:|------:|------:|-------:|-------:|-------:|
  | vie VTB | 3,874   | 16,399  | 4,963 | 21,029 | 2,537 | 9,591  |  
  | ind GSD | 19,260  | 75,181  | 4,568 | 11,318 | 4,348 | 10,495 |
  | eng EWT | 22,069  | 132,497 | 5,899 | 20,163 | 5,961 | 19,913 |

  The graphs are saved into `dat/dep/*.tsv` files. 

2. Run `network.ipynb` to read a graph and produce different kinds of nodes embeddings.

For directed graph:
  - in-degree centrality
  - eigenvector centrality
  - pagerank score

For undirected graph:
  - TODO 