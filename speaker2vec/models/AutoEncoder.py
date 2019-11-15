
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, d=None):
        super(AutoEncoder, self).__init__()
        if d is None:
            raise ValueError("input frames not specified")
        self.encoder1 = nn.Linear(in_features=d, out_features=4000)
        self.encoder2 = nn.Linear(in_features=4000, out_features=2000)
        self.embedding = nn.Linear(in_features=2000, out_features=256)
        self.decoder1 = nn.Linear(in_features=256, out_features=2000)
        self.decoder2 = nn.Linear(in_features=2000, out_features=4000)

    def forward(self, x):
        x = self.relu(self.encoder1(x))
        x = self.relu(self.encoder2(x))
        x = self.relu(self.embedding(x))
        x = self.relu(self.decoder1(x))
        o = self.decoder2(x)
        return o
