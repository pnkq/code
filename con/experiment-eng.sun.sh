#!/bin/bash

for k in 1 2 3
do
  bloop run -p con -m vlp.dep.DEP -- -l eng -t t+p -m predict -w 32 -h 200 -b 384
  bloop run -p con -m vlp.dep.DEP -- -l eng -t tg+p -m predict -h 256 -b 384
  bloop run -p con -m vlp.dep.DEP -- -l eng -t tn+p -m predict -h 256 -b 384
  bloop run -p con -m vlp.dep.DEP -- -l eng -t b -m predict -w 64 -h 200 -u 8 -b 384
done

