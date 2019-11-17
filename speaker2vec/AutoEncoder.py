import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, d=None, dropout=0.2):
        super(AutoEncoder, self).__init__()
        if d is None:
            raise ValueError("input feature size not specified")

        # fully connected layers
        self.enc1 = nn.Linear(in_features=d, out_features=6000)
        self.enc2 = nn.Linear(in_features=6000, out_features=2000)
        self.enc3 = nn.Linear(in_features=2000, out_features=256)
        self.dec1 = nn.Linear(in_features=256, out_features=2000)
        self.dec2 = nn.Linear(in_features=2000, out_features=6000)
        self.dec3 = nn.Linear(in_features=6000, out_features=d)

        # batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=6000)
        self.bn2 = nn.BatchNorm1d(num_features=2000)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.bn4 = nn.BatchNorm1d(num_features=2000)
        self.bn5 = nn.BatchNorm1d(num_features=6000)

        # non-linearity and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        # initialize weights
        nn.init.xavier_normal_(self.enc1.weight)
        nn.init.xavier_normal_(self.enc2.weight)
        nn.init.xavier_normal_(self.enc3.weight)
        nn.init.xavier_normal_(self.dec1.weight)
        nn.init.xavier_normal_(self.dec2.weight)
        nn.init.xavier_normal_(self.dec3.weight)

    def forward(self, x):
        # input, output tensor shapes: (batch_size, n_mfcc, n_frames)
        batch_size = x.shape[0]
        n_mfcc = x.shape[1]
        n_frames = x.shape[2]

        # 2d -> 1d
        x = torch.reshape(x, (batch_size, -1))

        # forward
        x = self.dropout(self.relu(self.bn1(self.enc1(x))))
        x = self.dropout(self.relu(self.bn2(self.enc2(x))))
        x = self.dropout(self.relu(self.bn3(self.enc3(x))))
        x = self.dropout(self.relu(self.bn4(self.dec1(x))))
        x = self.dropout(self.relu(self.bn5(self.dec2(x))))
        o = self.dec3(x)

        # 1d -> 2d
        o = torch.reshape(o, (batch_size, n_mfcc, n_frames))

        return o
