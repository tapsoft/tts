
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, d=None):
        super(AutoEncoder, self).__init__()
        if d is None:
            raise ValueError("input feature size not specified")

        # fully connected layers
        self.enc1 = nn.Linear(in_features=d, out_features=2000)
        self.enc2 = nn.Linear(in_features=2000, out_features=40)
        self.dec1 = nn.Linear(in_features=40, out_features=2000)
        self.dec2 = nn.Linear(in_features=2000, out_features=d)

        # initialize weights
        nn.init.xavier_normal_(self.enc1.weight)
        nn.init.xavier_normal_(self.enc2.weight)
        nn.init.xavier_normal_(self.dec1.weight)
        nn.init.xavier_normal_(self.dec2.weight)

    def forward(self, x):
        # input tensor shape: (batch_size, n_mfcc, n_frames)
        # output tensor shape: (batch_size, n_mfcc, n_frames)
        batch_size = x.shape[0]
        n_mfcc = x.shape[1]
        n_frames = x.shape[2]

        x = torch.reshape(x, (batch_size, -1))

        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.dec1(x))
        o = self.dec2(x)

        o = torch.reshape(o, (batch_size, n_mfcc, n_frames))

        return o
