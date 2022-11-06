
from torch import nn
from models.blocks import *


class CRNNmicro(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dsconv = nn.Sequential(
            DSConv2d(in_channels=1, out_channels=config.cnn_out_channels,
                     kernel_size=config.kernel_size, stride=config.stride),
            nn.Flatten(start_dim=1, end_dim=2),
        )

        self.conv_out_frequency = (config.n_mels - config.kernel_size[0]) // \
            config.stride[0] + 1

        self.gru = nn.GRU(
            input_size=self.conv_out_frequency * config.cnn_out_channels,
            hidden_size=config.hidden_size,
            num_layers=config.gru_num_layers,
            dropout=0.1,
            bidirectional=config.bidirectional,
            batch_first=True
        )

        self.attention = Attention(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input):
        input = input.unsqueeze(dim=1)
        conv_output = self.dsconv(input).transpose(-1, -2)
        gru_output, _ = self.gru(conv_output)
        contex_vector = self.attention(gru_output)
        output = self.classifier(contex_vector)
        return output
