
import os
import sys
import time
import math
#import wavio
import argparse
#import queue
import random
#import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
#import numpy as np
#import matplotlib.pyplot as plt
#from loader import *
#from models import AutoEncoder

file_paths = []
FILE_PATHS = "./file_paths.txt"


def get_pairs():
    pass


def get_distance():
    pass


def train():
    pass


def evaluate():
    pass


def make_paths():
    print("constructing file paths")
    # list of audio data paths
    for i in range(4):
        base_folder = '/data/KsponSpeech_0' + str(i + 1)
        for j in range(124):
            folder_index = str(124 * i + j + 1)
            folder = base_folder + '/KsponSpeech_' + '0' * (4 - len(folder_index)) + folder_index
            for k in range(1000):
                file_index = str((124 * i + j) * 1000 + k + 1)
                filename = '/KsponSpeech_' + '0' * (6 - len(file_index)) + file_index + '.pcm'
                file_paths.append(folder + filename)
    for j in range(497, 623):
        index = str(j)
        folder = '/data/KsponSpeech_05/KsponSpeech_0' + str(j)
        for k in range(1000):
            file_index = str((j - 1) * 1000 + k + 1)
            filename = '/KsponSpeech_' + file_index + '.pcm'
            file_paths.append(folder + filename)
    for k in range(622001, 622545):
        file_index = str(k)
        folder = '/data/KsponSpeech_05/KsponSpeech_0623'
        filename = '/KsponSpeech_' + file_index + '.pcm'
        file_paths.append(folder + filename)

    f = open(FILE_PATHS, "w")
    for file_path in file_paths:
        f.write(file_path+'\n')
        f.close()


def import_paths():
    print("loading file paths")
    f = open(FILE_PATHS, "r")
    lines = f.readlines()
    for line in lines:
        file_paths.append(line)
    f.close()


def split_dataset():
    pass


def main():
    if not os.path.isfile(FILE_PATHS):
        make_paths()
    else:
        import_paths()


if __name__ == "__main__":
    main()

