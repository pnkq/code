Rainfall Prediction using LSTM and BERT


1. Install JDK 1.8+, coursier, sbt and bloop:
    - coursier: https://get-coursier.io/docs/cli-installation
    - sbt: https://www.scala-sbt.org/
        `./cs setup` (where cs is the coursier script)
    - bloop: https://scalacenter.github.io/bloop/setup#universal
        `./cs install bloop --only-prebuilt=true`

2. Open CLI, cd to `code/s2s`, and invoke:

    sbt bloopInstall

    This will install all dependency libraries needed for the project.

3. Train a model:

    bloop run -p s2s -m s2s.Forecaster -- -m train -d simple -h 7 -l 7 -j 3 -r 256
    bloop run -p s2s -m s2s.Forecaster -- -m train -d complex -h 7 -l 7 -j 3 -r 512

    The result (performance and config) is saved to `data/result.jsonl`.

    If there is option `-w true` then the trained model is written out. You can open TensorBoard (sum/station) to see the training log.

    The trained model is saved to `bin/station/s` (if data is simple) or `bin/station/c` (if data is complex).

    tensorboard --logdir sum/station

    Using option `-t 2` for training a model which combines both LSTM and BERT. Days of a year are encoded by a BERT model
    and concatenated with a multilayer LSTM model for prediction.

    `-m train -d complex -t 2 -x 2 -y 2 -r 128 -i 16`

    Here, -x is the number of attention heads, -y is the number of encoder blocks, -r is the hidden size, and -i is the
    intermediate size (feed-forward size) of BERT. Note that both LSTM and BERT use -r hidden size for their encoder.


4. Evaluate a model:

    bloop run -p s2s -m s2s.Forecaster -- -m eval -d simple
    bloop run -p s2s -m s2s.Forecaster -- -m eval -d complex -s station

5. Arguments:

    -t model type (1 for LSTM, 2 for LSTM+BERT)
    -d simple/complex (data source)
    -m train/eval/experiment (mode)
    -s station name (viet-tri, vinh-yen,...)
    -h horizon (number of days)
    -l look back (number of days)
    -b batch size
    -k number of epochs
    -j number of recurrent layers (LSTM)
    -r hidden size (number of hidden units in each LSTM layer or in each BERT block)
    -p plot figures (false by default)
    -u bidirectional LSTM (false by default)
    -x number of attention heads (BERT)
    -y number of encoder blocks (BERT)
    -i intermediate size (ffn size of BERT)
    -x number of heads (BERT)
    -y number of blocks (BERT)

6. Run many experiments at a given station

    Use -m lstm
        bloop run -p s2s -m s2s.Forecaster -- -m lstm -d simple -s vinh-yen
        bloop run -p s2s -m s2s.Forecaster -- -m lstm -d complex -s viet-tri

    Use -m bert
        bloop run -p s2s -m s2s.Forecaster -- -m bert -d complex -s viet-tri

    Results are saved to `data/result-bert-viet-tri.jsonl`. Each line contains a model result.

6. Run many experiments at a given region

    Use -m lstm and simple data
        bloop run -p s2s -m s2s.Forecaster -- -m lstm -d clusterS -s tay-bac

