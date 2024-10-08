#!/bin/bash

# phuonglh for batch experiments with multiple languages and models
modelTypes=("t" "t+p" "f" "x")

for t in $modelTypes
do
  bloop run -p con -m vlp.dep.DEPx -- -l vie -t $t -m validate -J-Xmx8g
done
