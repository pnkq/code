LSTM-based Rainfall Prediction

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

    The trained model is saved to `bin/station/s` (if data is simple) or `bin/station/c` (if data is complex).

    You may want to open TensorBoard (sum/lstm) to see the training log.

    tensorboard --logdir sum/

4. Evaluate a model:

    bloop run -p s2s -m s2s.Forecaster -- -m eval -d simple

5. Arguments:

    -d simple/complex (data source)
    -m train/eval/experiment (mode)
    -s station name (viet-tri, vinh-yen,...)
    -h horizon (number of days)
    -l look back (number of days)
    -b batch size
    -k number of epochs
    -j numLayers (LSTM layers)
    -r recurrent size (number of hidden units in each LSTM layer)
    -p plot figures (false by default)
    -u bidirectional LSTM (false by default)

6. Run many experiments: -m experiment

    bloop run -p s2s -m s2s.Forecaster -- -m experiment -d simple -s S1
    bloop run -p s2s -m s2s.Forecaster -- -m experiment -d complex -s viet-tri