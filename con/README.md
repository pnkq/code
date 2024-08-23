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

2. Run `network.ipynb` to read a graph and produce different kinds of nodes embeddings. Each kind is in a separate file.

For directed graph:
  - in-degree centrality
  - eigenvector centrality
  - pagerank score

For undirected graph:
  - TODO 

3. Run `vlp.dep.Merger` to merge different embeddings into a (parquet) file. Each node is associated with a vector 
  of graph embeddings. 
  
4. Run `vlp.dep.DEPx` with the model type `x` (option `-t x`) to experiment with interested models. 
   Here, the features include (token embeddings, part-of-speech embeddings) and (graph embeddings). This is a multi-input BigDL graph model.
   In the inference stage, the `x` model needs to take as input an array of feature columns (`t+p` and `x`). 
   NOTE: in the implementation of `x` model, we intentionally split a long graph (whose length > config.maxSeqLen) into two halves left and 
   right using the ROOT token. Since we are concerned with projective dependency parsing, we assume that all the annotated graphs 
   in the treebanks are projective. 