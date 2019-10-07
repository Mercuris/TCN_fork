from torch import nn
from TCN.tcn import TemporalConvNet
import torch.nn.functional as F


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        # input_size = 88, output_size = 88, num_channels = [150] * 4, kernel_size = 5
        super(TCN, self).__init__()
        # With Batch size as N, Seq len as L and Number of features as C, initial data is N x L x C (C = 88)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # Output is N x L x C
        output = self.linear(output).double()
        return self.sig(output)
