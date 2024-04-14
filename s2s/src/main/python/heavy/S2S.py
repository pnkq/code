import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import sklearn
import sklearn.metrics

from os.path import expanduser
from bigdl.chronos.data import TSDataset
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.forecaster.seq2seq_forecaster import Seq2SeqForecaster
from torch import nn

def isHeavy(value):
    if value >= 50:
        return 1
    else:
        return 0
def readData(basePath, stationIndex):
    pathY = f"{basePath}y.80-19.tsv"
    dfY = pd.read_csv(pathY, delimiter="\t", header=None)
    dfY['Station'] = pd.to_numeric(dfY.loc[:, 3+stationIndex])
    dfY['Date'] = dfY.loc[:, 0:2].apply(lambda row: '-'.join(row.values.astype(str)), axis=1)
    dfY['Date'] = pd.to_datetime(dfY['Date'])
    target = dfY['Station'].apply(lambda v: isHeavy(v))
    dfY['y'] = target
    return dfY

def plot(p_valid, y_valid, modelName):
    plt.figure(figsize=(10, 4))
    # plt.plot(p_valid[:,:,0][:,-1][horizon:horizon+366]) # shift horizon time steps
    plt.plot(p_valid[:, :, 0][:, -1][horizon+366:horizon+732])  # 2nd year of the valid set
    plt.plot(y_valid[:, :, 0][:, -1][366:732])
    plt.legend(["prediction", "ground truth"])
    plt.title(modelName + ' Extreme Precipitation Prediction in 2013 at Station ' + str(stationIndex))
    plt.xlabel('Day')
    plt.ylabel('Normalized Predicted Value')
    plt.savefig(modelName + '-14-' + str(stationIndex) + ".png")

def classify(p, epsilon=0.5):
    score = 1 / (1 + np.exp(-p))
    if score >= epsilon:
        return 1.0
    else:
        return 0.0


home = expanduser("~")
stationIndex = 19
df = readData(home + "/manuscripts/code/s2s/dat/lnq/", stationIndex)
train, valid, test = TSDataset.from_pandas(df, dt_col='Date', target_col='y', extra_feature_col=[], with_split=True,
                                           val_ratio=0.1, test_ratio=0.1)
lookback, horizon = 7, 28
scaler = StandardScaler()
for data in [train, valid, test]:
    data.deduplicate() \
        .impute() \
        .gen_dt_feature() \
        .scale(scaler, fit=(data is train)) \
        .roll(lookback=lookback, horizon=horizon)

X, y = train.to_numpy()
print(X.shape, y.shape)

extreme_weight = torch.full([1], 9.0)  # make the extreme sample a large weight
loss = nn.BCEWithLogitsLoss(pos_weight=extreme_weight)

model = Seq2SeqForecaster(past_seq_len=lookback, future_seq_len=horizon, input_feature_num=X.shape[-1],
                          output_feature_num=y.shape[-1], optimizer='Adam', lstm_hidden_dim=32, lstm_layer_num=1,
                          loss=loss, lr=1e-5)
model.fit((X, y), batch_size=32, epochs=10)

X_valid, y_valid = valid.to_numpy()
p_valid_s2s = model.predict(X_valid)
X_test, y_test = test.to_numpy()
p_test_s2s = model.predict(X_test)

plot(p_test_s2s, y_test, "S2S")

y_v = valid.unscale_numpy(y_valid)[:, :, 0][:, -1][:-horizon]  # the whole valid set
z_v_s2s = [classify(p) for p in p_valid_s2s[:, :, 0][:, -1]][horizon:]
y_t = test.unscale_numpy(y_test)[:, :, 0][:, -1][:-horizon]  # the whole test set
z_t_s2s = [classify(p) for p in p_test_s2s[:, :, 0][:, -1]][horizon:]

# AUC score
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_v, z_v_s2s)
print(sklearn.metrics.auc(fpr, tpr))
print("VALID:\n")
print(sklearn.metrics.classification_report(y_v, z_v_s2s))
print("TEST:\n")
print(sklearn.metrics.classification_report(y_t, z_t_s2s))