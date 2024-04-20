#!/bin/bash

# phuonglh for batch experiments with multiple languages and models
# on SUN server, there are 24 cores. Each core processes 16 samples, thus the batch size is 16 x 24 = 384.
# -X 24 -b 384. Since Y = 4, there are 6 executors.

# languages="czech english german italian swedish"
languages="ind"
modelTypes="tg+p"
hiddenSizes="64 128 200 256 300"

for lang in $languages
do
  for modelType in $modelTypes
  do
    for h in $hiddenSizes
    do
      for k in 1 2 3
      do
        bloop run -p con -m vlp.dep.DEP -- -l $lang -m train -t $modelType -h $h -X 24 -b 384 -k 50
      done
    done
  done
done

