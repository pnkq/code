# Introduction

- Implementation of a joint method for Dialogue Act Detection (DAC).
- Empirical results on two standard benchmark dialogue datasets for English and Vietnamese

# Running

## Tools

1. Read, transform and save datasets: `woz.DialogReader` and `woz.nlu.NLU -- -m init`
2. Main module `woz.nlu.NLU`

## Models

**LSTM**

| j | maxSeqLen | w | r | h | k | trainingTime | speed | devAcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20 | 32 | 64 | 64 | 100 | 10.41 hours | 280 samples/second | 0.9211 | 


# Statistics

Frequency of numbers of acts in each turn:

|  n|count|
|---|-----|
|  1|10178|
|  3|  307|
|  4|    8|
|  2| 4101|
|  0|  154|
