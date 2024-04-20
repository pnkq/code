LSTM-based Rainfall Prediction


1. Install JDK and SBT

2. Open CLI, cd to code/s2s, and invoke:

    sbt bloopInstall

3. Train a model:

    bloop run -p s2s -m s2s.Forecaster -- -m train -d simple -h 7 -l 7 (parameters)

    Can open TensorBoard (sum/lstm) to see the training log.

    tensorboard --logdir sum/lstm

4. Evaluate a model:

    bloop run -p s2s -m s2s.Forecaster -- -m eval -d simple

