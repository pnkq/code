# Introduction

- Implementation of a joint method for Dialogue Act Detection (DAC).
- Empirical results on two standard benchmark dialogue datasets for English and Vietnamese

# Running

## Tools

1. Read, transform and save datasets: `woz.DialogReader` and `woz.nlu.NLU -- -m init`
2. Main module `woz.nlu.NLU` with options -t lstm/bert

## Single Models

**LSTM**

| j | maxSeqLen | w | r | h | k | trainingTime | speed              | devAcc |
|---| --- | --- | --- | --- | --- |--------------|--------------------|--------|
| 1 | 20 | 32 | 64 | 64 | 100 | 10.41 hours  | 280 samples/second | 0.9211 | 
| 2 | 20 | 32 | 64 | 64 | 100 | 13.40 hours  | 230 samples/second | 0.9068 |


**BERT**

| j | maxSeqLen | w | r | h | k | trainingTime | speed              | devAcc |
|---| --- | --- | --- | --- | --- |--------------|--------------------|--------|
| 1 | 20 | 32 | 64 | 64 | 100 | ? hours      | ? samples/second   | 0.9211 | 
| 2 | 20 | 32 | 64 | 64 | 100 | 13.54 hours  | 200 samples/second | 0.8740 |

## Joint Models

1. Input is a sequence of tokens: `[t1, t2,..., tN]`;
2. Embed this sequence by an `Embedding` layer to get a sequence of embedding vectors: `[w1, w2,..., wN]`;  
3. Pass this sequence to a (possibly multilayer) bidirectional encoder to get a sequence of state vectors `[h1, h2,..., hN]`;
4. Split into 2 branches:
  a) Pass state vectors to a `Dense(numEntities, softmax)` layer to get output sequence `[o1, o2, ..., oN]`.
  b) Get the last state `hN`, pass it to a `Dense(numActs, sigmoid)` layer to get an output vector `a`.
  c) Duplicate `a` to have another output vector `a`
5. Combine 4a) and 4c) into a sequence of (N+2) vectors: `[o1, o2, ..., oN, a, a]`. However, we use a clever trick to get the NLL criterion work as follows:
  a) Right pad each vector `o` by `numActs` zero.
  b) Left pad each vector `a` by `numEntities` zero and concat them to the end of `o` sequence.
  This will result in a sequence of (N+2) vectors, each has a dimension of (numEntities + numActs) elements. 
6. In the target vector, we need to shift the act indices by `numEntities` for the negative log-likelihood (NLL) training criterion to work properly.

In the joint model:
- The feature size is `Array(maxSeqLen)` as in a single model.
- The label size is `Array(maxSeqLen + 2)` where `Array(2)` represents a target act vector. 

# Statistics

Frequency of numbers of acts in each turn:

|  n|count|
|---|-----|
|  1|10178|
|  3|  307|
|  4|    8|
|  2| 4101|
|  0|  154|

Frequency of lengths of each turn (in tokens) in the validation corpus:

|  m| count |
|---|-------|
| 10| 996   |
|  7| 918   |
|  8| 910   |
|  9| 894   |
|  6| 892   |
| 11| 892   |
| 12| 831   |
| 13| 767   |
| 14| 727   |
| 15| 689   |
|  5| 675   |
| 16| 604   |
| 17| 580   |
| 19| 474   |
| 18| 465   |
| 20| 416   |
|  4| 379   |
| 21| 375   |
| 22| 357   |
| 23| 324   |
| 24| 255   |
| 25| 234   |
| 27| 190   |
|  3| 183   |
| 26| 180   |
| 28| 167   |
| 29| 147   |
| 30| 114   |
| 31| 68    |
| 32| 19    |
|  2| 16    |
| 33| 2     |
| 34| 2     |
|  1| 1     |
| 35| 1     |
| 41| 1     |
| 42| 1     |
| 38| 1     |
| 36| 1     |

There are 12,309 samples whose length <= 20, which accounts for 12,309/14,748 = 83.46\%. If we use the maxSeqLen=25, the ratio is 93.94\%.

# TODO

- Implement act inference (sigmoid threshold of 0.5)
- Build global act graph
- Learn act node embeddings over the global act graph
- Start manuscript (sentence embedding models)

# DONE
- Add word shapes as important features for slot detection
- Export tagging results
- Implement the joint model
