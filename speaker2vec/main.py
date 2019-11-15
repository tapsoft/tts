
import os
import sys
import math
import time
import argparse
import queue
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from loader import *
from models import AutoEncoder

FILE_PATHS = "~/zeroshot-tts-korean/file_paths.txt"
file_paths = []


def get_pairs():
    pass


def get_distance():
    pass


def train():
    pass


def evaluate():
    pass


def import_paths():
    print("loading file paths")
    f = open(FILE_PATHS, "r")
    lines = f.readlines()
    for line in lines:
        file_paths.append(line[:-1])
    f.close()

    return None


def split_dataset(batch_size=8, valid_ratio=0.01, num_workers=2):
    train_loader_count = num_workers
    audio_num = len(file_paths)
    batch_num = math.ceil(audio_num / batch_size)

    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num

    batch_num_per_train_loader = math.ceil(train_batch_num/num_workers)

    train_begin = 0
    train_end_raw_id = 0
    train_dataset_list = list()

    for i in range(num_workers):
        train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

        train_begin_raw_id = train_begin * batch_size
        train_end_raw_id = train_end * batch_size

        train_dataset_list.append(BaseDataset(
            file_paths[train_begin_raw_id:train_end_raw_id],
            train_mode=True))

        train_begin = train_end

    valid_dataset = BaseDataset(file_paths[train_end_raw_id:],train_mode=False)

    return train_batch_num, train_dataset_list, valid_dataset


def main():

    feature_size = 4000
    learning_rate = 1e-4
    num_workers = 2
    batch_size = 8
    max_epochs = 2

    # set random seed
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda')

    # initialize model
    model = AutoEncoder(d=feature_size)
    model.flatten_parameters()

    model = nn.DataParallel(model).to(device)

    optimizer = optim.Adam(model.module.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='mean').to(device)

    # import audio file paths
    import_paths()
    for i in range(10):
        print(file_paths[i])

    best_loss = 1e20
    begin_epoch = 0

    train_batch_num, train_dataset_list, valid_dataset = split_dataset(
        batch_size=batch_size, valid_ratio=0.01, num_workers=num_workers)

    logger.info('start')

    train_begin = time.time()
    for epoch in range(begin_epoch,max_epochs):
        print("epoch", epoch)
        train_queue = queue.Queue(num_workers * 2)
        train_loader = MultiLoader(train_dataset_list, train_queue, batch_size, num_workers)
        train_loss = train(model, train_batch_num, train_queue,
                           criterion, optimizer, device, train_begin, num_workers, 10)
        logger.info("Epoch %d Training Loss %0.4f" % (epoch, train_loss))
        train_loader.join()

    valid_queue = queue.Queue(num_workers * 2)
    valid_loader = BaseDataLoader(valid_dataset, valid_queue, batch_size, 0)
    valid_loader.start()
    print("start eval")
    eval_loss = evaluate(model, valid_loader, valid_queue,
                         criterion,device)
    valid_loader.join()
    print("end eval")

    # save every epoch
    save_name = "model_%03d" %(epoch)
    # save best loss model
    is_best_loss = (eval_loss < best_loss)
    if is_best_loss:
        best_loss = eval_loss

    return 0


if __name__ == "__main__":
    main()

