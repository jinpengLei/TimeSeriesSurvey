import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        bidirectional = False
        if configs.binary == 2:
            bidirectional = True
        print(configs)
        self.input_size = configs.input_size
        self.num_hiddens = configs.hidden_size
        self.output_size = configs.output_size
        self.lstm_layer = nn.LSTM(configs.input_size, configs.hidden_size, batch_first=True, bidirectional=bidirectional)
        self.output_size = configs.output_size

        if not self.lstm_layer.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.output_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)

    def forward(self, X):
        X = X.to(torch.float32)
        o, (hid, cell) = self.lstm_layer(X)
        output = self.linear(hid.reshape((-1,  hid.shape[-1])))
        return output.unsqueeze(0)