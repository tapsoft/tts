
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

FILE_PATHS = "D:/GitHub_Repos/zeroshot-tts-korean/file_paths.txt"
# FILE_PATHS = "~/zeroshot-tts-korean/file_paths.txt"
file_paths = []


def train(model, total_batch_size, queue, criterion, optimizer, device, train_begin, train_loader_count):
    print_batch = 5

    total_loss = 0.
    total_num = 0
    batch = 0

    # set model to train mode
    model.train()

    logger.info('train() start')
    begin = epoch_begin = time.time()

    while True:
        if queue.empty():
            logger.debug('queue is empty')

        # input, target tensor shapes: (batch_size, n_mfcc, n_frames)
        inputs, targets = queue.get()
        batch_size = inputs.shape[0]

        if inputs.shape[0] == 0:
            # empty feats means closing one loader
            train_loader_count -= 1
            logger.debug('left train_loader: %d' % (train_loader_count))

            # if all loader closed, stop training
            if train_loader_count == 0:
                break
            else:
                continue

        # flush gradients
        optimizer.zero_grad()

        inputs = inputs.to(device)
        targets = targets.to(device)

        # output tensor shape: (batch_size, n_mfcc, n_frames)
        # compute output
        model.module.flatten_parameters()
        output = model(inputs).to(device)

        loss = criterion(output.contiguous().view(-1), targets.contiguous().view(-1))
        total_loss += loss.item()
        total_num += batch_size

        loss.backward()
        optimizer.step()

        if batch % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('batch: {:4d}/{:4d}, batch size: {:3d} loss: {:.4f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'
                        .format(batch,
                                total_batch_size,
                                batch_size,
                                total_loss / total_num,
                                elapsed, epoch_elapsed, train_elapsed))
            begin = time.time()

        batch += 1
        train.cumulative_batch_count += 1

    logger.info('train() completed')
    return total_loss / total_num


def evaluate():
    print("eval step. passing")
    return 0


def import_paths():
    print("loading file paths")
    f = open(FILE_PATHS, "r")
    lines = f.readlines()
    for line in lines:
        file_paths.append(line[:-1])
    f.close()


def split_dataset(batch_size=32, valid_ratio=0.01, num_workers=2):
    # split train/val dataset
    # construct BaseDataset objects using file_paths
    # train_batch_num: number of batches in training data
    # train_dataset_list: list of BaseDataset objects of training data
    # val_dataset: BaseDataset object of validation data
    train_loader_count = num_workers
    file_num = len(file_paths)
    batch_num = math.ceil(file_num / batch_size)

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

        train_dataset_list.append(
            BaseDataset(file_paths[train_begin_raw_id:train_end_raw_id], train_mode=True))

        train_begin = train_end

    valid_dataset = BaseDataset(file_paths[train_end_raw_id:], train_mode=False)

    return train_batch_num, train_dataset_list, valid_dataset


def main():
    feature_size = 4000
    learning_rate = 1e-4
    num_workers = 2
    batch_size = 32
    max_epochs = 2

    # set random seed
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # set device to gpu if available
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    # initialize model
    model = AutoEncoder(d=feature_size)
    model.flatten_parameters()

    # load model to device
    model = nn.DataParallel(model).to(device)

    # set optimizer and loss
    optimizer = optim.Adam(model.module.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='sum').to(device)

    # import data file paths
    import_paths()
    for i in range(10):
        print(file_paths[i])
    # randomly shuffle data
    random.shuffle(file_paths)

    best_loss = 1e20
    begin_epoch = 0

    # split training and validation data
    train_batch_num, train_dataset_list, valid_dataset = split_dataset(batch_size=batch_size, valid_ratio=0.01, num_workers=num_workers)

    # begin logging
    logger.info('start')

    train_begin = time.time()
    for epoch in range(begin_epoch, max_epochs):
        print("epoch", epoch)

        train_queue = queue.Queue(num_workers * 2)
        train_loader = MultiLoader(train_dataset_list, train_queue, batch_size, num_workers)
        train_loss = train(model, train_batch_num, train_queue, criterion, optimizer, device, train_begin, num_workers, 10)

        logger.info("Epoch %d Training Loss %0.4f" % (epoch, train_loss))
        train_loader.join()

        valid_queue = queue.Queue(num_workers * 2)
        valid_loader = BaseDataLoader(valid_dataset, valid_queue, batch_size, 0)
        valid_loader.start()

        print("start eval")
        eval_loss = evaluate(model, valid_loader, valid_queue, criterion,device)
        valid_loader.join()
        print("end eval")

        # save every epoch
        save_name = "model_%03d" % epoch

        # save best loss model
        is_best_loss = (eval_loss < best_loss)
        if is_best_loss:
            print("best model: " + save_name)
            best_loss = eval_loss

    return 0


if __name__ == "__main__":
    main()

