import torch
import torchaudio
from torch import nn
from models.blocks import *


class CRNNMicroStreaming(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_window_len = 101  # = 4 sec
        self.hidden = torch.Tensor(0)
        self.init_buffer()

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
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=config.n_mels
        ).to(config.device)

    def forward(self, input):
        spectrogram = torch.log(self.melspec(input).clamp_(
            min=1e-9, max=1e9)).squeeze(0)
        self.buffer = torch.cat([self.buffer, spectrogram], dim=1)[
            :, -self.max_window_len:]
        spectrogram = self.buffer.unsqueeze(dim=0).unsqueeze(dim=0)

        conv_output = self.dsconv(spectrogram).transpose(-1, -2)
        hidden = self.hidden if len(self.hidden.size()) > 1 else None
        gru_output, self.hidden = self.gru(conv_output, hidden)

        contex_vector = self.attention(gru_output)
        output = self.classifier(contex_vector)
        return nn.functional.softmax(output, dim=1)[0][1]

    def init_buffer(self):
        self.buffer = torch.zeros(size=(40, self.max_window_len))
