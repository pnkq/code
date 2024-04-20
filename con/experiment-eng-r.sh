#!/bin/bash

# g and n model types: try with different hidden dimension, the word embedding size is always 300.
#
modelTypes="t+p"
embeddingSizes="32 64 100"
hiddenSizes="32 64 128 200 256 300"

for lang in $languages
do
  for w in $embeddingSizes
    for h in $hiddenSizes
    do
      for k in 1 2 3
        bloop run -p con -m vlp.dep.DEP -- -l eng -m train -t t+p -w $w -h $h -b 32
      end
    done
  done
done

