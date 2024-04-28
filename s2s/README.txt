LSTM-based Rainfall Prediction


1. Install JDK 1.8, SBT and BLOOP.

2. Open CLI, cd to `code/s2s`, and invoke:

    sbt bloopInstall

    This will install all dependent libraries.

3. Train a model:

    bloop run -p s2s -m s2s.Forecaster -- -m train -d simple -h 7 -l 7 -j 3 -r 256
    bloop run -p s2s -m s2s.Forecaster -- -m train -d complex -h 7 -l 7 -j 3 -r 512

    The trained model is saved to `bin/station/s` (if data is simple) or `bin/station/c` (if data is complex).

    You may want to open TensorBoard (sum/lstm) to see the training log.

    tensorboard --logdir sum/

4. Evaluate a model:

    bloop run -p s2s -m s2s.Forecaster -- -m eval -d simple

5. Arguments:

    -d simple/complex (data source)
    -m train/eval (mode)
    -s stationName (viet-tri, vinh-yen,...)
    -h horizon (number of days)
    -l look back (number of days)
    -b batchSize
    -k number of epochs
    -j numLayers (LSTM layers)
    -r recurrentSize (number of hidden units in each LSTM layer)
    -p plot figures (false by default)
    -u bidirectional LSTM (false by default)
