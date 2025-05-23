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

7. Run many experiments at a given region

    Use -m lstm and simple data
        bloop run -p s2s -m s2s.Forecaster -- -m lstm -d clusterS -s tay-bac
    Use -m lstm and complex data
        bloop run -p s2s -m s2s.Forecaster -- -m lstm -d clusterC -s tay-bac
    Use -m bert and complex data
        bloop run -p s2s -m s2s.Forecaster -- -m bert -d clusterC -s tay-bac

8. Evaluate a model on a data set

    bloop run -p s2s -m s2s.Forecaster -- -m eval -s dong-bac -d complexC -t 1


8.1 Northeast:

    dong-bac, modelType=1, r=128 hidden units, j=2 layers.
        avg(MAE) = [0.5819589414629818,0.565427281436763,0.6153264698854685,0.611238412742756,0.5812697864527732,0.5857255701465901,0.5753226863314148]
        avg(MSE) = [1.3085980953648695,1.2849396170712002,1.3564906686345564,1.3587250509808328,1.3015439738973635,1.3051810527228742,1.2466166082326]

    Best config: r=512, a=2, b=4.

    h=7, single:
        avg(MAE) = [0.6145047112274078,0.6107021507210396,0.6049472158288463,0.5478134701898565,0.5422360489032937,0.5560494454218682,0.5866176700623735]
        avg(MSE) = [0.9515238214007586,0.9595362464974097,0.9664359502858418,1.0196677525236524,1.0147966685565848,1.011404378879273,0.9739454350936344]

    h=7, dual:
    
8.2 North:

    h=7, single: 
        avg(MAE) = [0.6985653151843298,0.6208457168611072,0.7319037516674366,0.64420156588092,0.6790197447498004,0.6715568916379131,0.6052586209849707]
        avg(MSE) = [1.0871708850217408,1.135301962674293,1.091754812292869,1.0999446260112378,1.091528453149621,1.0852837836268157,1.1788239199113824]
    h=7, dual: 

8.3 Northwest

    h=7, single:
        