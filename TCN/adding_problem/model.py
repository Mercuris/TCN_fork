from torch import nn
from TCN.tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        # input_size = 2, output_size = 1, num_channels = [nhid] * levels = [30] * 4, kernel_size = 7
        super(TCN, self).__init__()
        # With Batch size as N, Seq len as L and 2 features, initial data is N x 2 x L
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # input to linear is N x nhid x l = N x 30 x 1 where l is last element of sequence
        self.linear = nn.Linear(num_channels[-1], output_size)
        # output of linear is N x 1 x 1
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])
