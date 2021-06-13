import torch
import torch.nn as nn


class BiLSTMDetector(nn.Module):
    def __init__(self, lstm_hidden_dim=64, lstm_layers_count=1, conv_out_channels=10, fc_hidden_dim=1024):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, conv_out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.bi_lstm = nn.LSTM(input_size=conv_out_channels, hidden_size=lstm_hidden_dim, num_layers=lstm_layers_count,
                               proj_size=0, bidirectional=True)
        self.fc1 = nn.Linear(lstm_hidden_dim * 2, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, 1)

    def forward(self, inputs):
        inputs = inputs.transpose(2,1)
        output = self.encoder(inputs)
        output = output.permute(2, 0, 1)
        output, (hn, cn) = self.bi_lstm(output)
        output = torch.relu(self.fc1(output))
        output = self.fc2(output)
        output = output.permute(1, 0, 2)
        return output