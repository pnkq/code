LSTM-based Rainfall Prediction


bloop run -p s2s -m s2s.Forecaster -- -m train -d simple -h 7 -l 7
bloop run -p s2s -m s2s.Forecaster -- -m eval -d simple

