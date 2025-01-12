# Introduction

- Implementation of a joint method for Dialogue Act Detection (DAC).
- Empirical results on two standard benchmark dialogue datasets for English and Vietnamese

# Running

## Tools

1. Read, transform and save datasets: `woz.DialogReader` and `woz.nlu.NLU -- -m init`
2. Main module `woz.nlu.NLU`

## Models

    LSTM, 
    j=1, 
    maxSeqLen=20, 
    w=32, 
    r=64, 
    h=64, 
    k=100, 
    trainingTime=10.41 hours (DELL), 280 samples/second
    devAcc=0.9211

