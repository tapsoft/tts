
import os
import sys
import math
import wavio
import time
import torch
import random
import threading
import logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
from preprocessing import trim

PAD = 0
N_FFT = 512
SAMPLE_RATE = 16000

target_dict = dict()

def load_targets():
    pass


def get_mfcc_feature():
    pass

class BaseDataset(Dataset):
    def __init(self):
        pass

    def __len__(self):
        pass

    def count(self):
        pass

    def getitem(self):
        pass

