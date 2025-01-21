import pandas as pd, numpy as np, os, logging, sys, inspect, itertools, datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

from .tsfCommon import *


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out


class modelPyTorch:
    __all__ = ['modelPyTorchLSTM']

    def sliding_windows(self, data):
        x = []
        for i in data:
            x.append(pd.DataFrame(i).iloc[:, :].values.tolist())
        return np.array(x)

    def runModelPyTorchLSTM(self, x_train_R, x_test_R, y_train_R):
        try:

            num_epochs = 1000 + round(abs(1000 - (x_train_R.shape[1] + y_train_R.shape[0])))
            learning_rate = 0.01
            input_size = 1
            hidden_size = 2
            num_layers = 1
            num_classes = 1

            scaler = MinMaxScaler()
            x_train = Variable(torch.Tensor(np.array(self.sliding_windows(scaler.fit_transform(x_train_R)))))
            x_test = Variable(torch.Tensor(
                np.array(self.sliding_windows(scaler.fit_transform(x_test_R.reset_index().iloc[:, 1:].values)))))
            y_train = Variable(torch.Tensor(np.array(scaler.fit_transform(y_train_R.reset_index().iloc[:, 1:].values))))
            lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
            for epoch in range(num_epochs):
                outputs = lstm(x_train)
                optimizer.zero_grad()
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            logging.info("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
            lstm.eval()
            return scaler.inverse_transform(lstm(x_test).data.numpy())

        except Exception as e:
            print(e)

    def modelPyTorchLSTM(self, seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach):
        rtVal = []
        try:
            modelTest = self.runModelPyTorchLSTM(x_train, x_test, y_train)
            pred = pd.concat([y_test.reset_index(), pd.DataFrame(modelTest, columns=['predicted_mean'])], axis=1)
            modelFur = self.runModelPyTorchLSTM(pd.concat([x_train, x_test], axis=0), furDF,
                                                pd.concat([y_train, y_test], axis=0))
            furData = pd.concat([y_test.reset_index(), pd.DataFrame(modelFur, columns=['predicted_mean'])], axis=1)
            modelSt = tsfCommon().modelEval(pred['Data'], pred['predicted_mean'])
            rtVal.append(
                ["PyTorch", "Long short-term memory (LSTM)", True, modelSt[0], modelSt[1], modelSt[2], modelSt[3],
                 modelSt[4], modelSt[5], modelSt[6], modelSt[7], modelSt[8], modelSt[9],
                 pred[['Date', 'predicted_mean']].values.tolist(), furData[['Date', 'predicted_mean']].values.tolist()])
        except Exception as e:
            rtVal.append(["PyTorch", "Long short-term memory (LSTM)", False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], []])
            print(e)

        return rtVal
