from torch import nn
from TCN.tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        # input_size = 1, output_size = 10, num_channels = [nhid] * levels = [10] * 8, kernel_size = 8
        super(TCN, self).__init__()
        # With Batch size as N, Seq len as L and 1 feature, initial data is 32 x 1 x 1020
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # input to linear is N x L x nhid = 32 x 1020 x 10
        self.linear = nn.Linear(num_channels[-1], output_size)
        # input of linear is N x L x nhid = 32 x 1020 x 10
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        y2 = y1.transpose(1, 2)
        out = self.linear(y2)
        # return self.linear(y1.transpose(1, 2))
        return out
