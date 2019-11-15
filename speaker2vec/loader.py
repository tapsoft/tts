
import os
import sys
import math
import time
import torch
import random
import threading
import logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
from preprocessing import trim
from wavio import readwav

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

PAD = 0
N_FFT = 512
SAMPLE_RATE = 16000

target_dict = dict()


def get_mfcc_feature(filepath):
    (rate, width, sig) = readwav(filepath)
    sig = sig.ravel()
    sig = trim(sig)


class BaseDataset(Dataset):
    def __init(self):
        pass

    def __len__(self):
        pass

    def count(self):
        pass

    def getitem(self):
        pass

