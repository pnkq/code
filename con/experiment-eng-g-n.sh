#!/bin/bash

# g and n model types: try with different hidden dimension, the word embedding size is always 300.
#
modelTypes="tg+p tn+p"
hiddenSizes="32 64 128 200 256 300"

do
  for modelType in $modelTypes
  do
    for h in $hiddenSizes
    do
      for k in 1 2 3
        bloop run -p con -m vlp.dep.DEP -- -l eng -m train -t $modelType -h $h -b 32
      end
    done
  done
done

