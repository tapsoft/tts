
import os
import sys
import time
import math
#import wavio
import argparse
#import queue
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
#from loader import *
#from models import AutoEncoder

file_paths = []

def get_pairs():
    pass


def get_distance():
    pass


def train():
    pass


def evaluate():
    pass


def make_dataset():
    # list of audio data paths
    for i in range(4):
        path = []
        base_folder = '/data/KsponSpeech_0' + str(i + 1)
        for j in range(124):
            folder_index = str(124 * i + j + 1)
            folder = base_folder + '/KsponSpeech_' + '0' * (4 - len(folder_index)) + folder_index
            for k in range(1000):
                file_index = str((124 * i + j) * 1000 + k + 1)
                filename = 'KsponSpeech_' + '0' * (6 - len(file_index)) + file_index + '.pcm'
                file_paths.append([folder, filename])
    for j in range(497, 623):
        index = str(j)
        folder = '/data/KsponSpeech_05/KsponSpeech_0' + str(j)
        for k in range(1000):
            file_index = str((j - 1) * 1000 + k + 1)
            filename = 'KsponSpeech_' + file_index + '.pcm'
            file_paths.append([folder, filename])
    for k in range(622001, 622545):
        file_index = str(k)
        folder = '/data/KsponSpeech_05/KsponSpeech_0623'
        filename = 'KsponSpeech_' + file_index + '.pcm'
        file_paths.append([folder, filename])

    print(file_paths)

    # Read Data
    '''
    transcripts = []
    total_english_words = []
    error_files = []
    for i in range(5):
        for file in file_paths[i]:
            folder, filename = file[0], file[1]
            result = check_file_text(folder + '/' + filename)

            if result[0] == 'parsed':
                processed_line, english_words = result[1], result[2]
                # transcripts.append([filename, processed_line])
                transcripts.append([folder + '/' + filename.replace('.txt', '.pcm'), processed_line])
                if len(english_words) > 0:
                    total_english_words.append([filename, english_words])
            else:
                filename, ex = result[1], result[2]
                error_files.append([filename, ex])
        print('Reading folder ' + str(i + 1) + ' is finished')
        '''


def main():
    make_dataset()


if __name__ == "__main__":
    main()

