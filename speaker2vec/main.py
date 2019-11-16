import os
import sys
import math
import time
import queue
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import matplotlib.pyplot as plt
from loader import *
from AutoEncoder import AutoEncoder

# training data location
# FILE_PATHS = "D:/GitHub_Repos/zeroshot-tts-korean/file_paths.txt"
FILE_PATHS = "/home/cs470/zeroshot-tts-korean/file_paths.txt"
file_paths = []
SAVE_PATH = "/home/cd470/zeroshot-tts-korean/speaker2vec/checkpoints/"

# preprocessing
n_mfcc = 40
n_frames = 100

# hyperparameters
max_epochs = 10
batch_size = 32
learning_rate = 1e-4
valid_ratio = 0.01
num_workers = 2


def train(model, total_batch_size, queue, criterion, optimizer, device, train_begin, train_loader_count, print_batch=5):
    print_loss = 0.
    total_loss = 0
    batch = 0

    # set model to train mode
    model.train()

    # begin logging
    logger.info('train() start')
    begin = epoch_begin = time.time()

    while True:
        batch += 1

        if queue.empty():
            logger.debug('queue is empty')

        # input, target tensor shapes: (batch_size, n_mfcc, n_frames)
        inputs, targets = queue.get()
        batch_size = inputs.shape[0]

        logger.debug('batch %d, queue length: %d, got a tensor of shape (%d, %d, %d)'
                     % (batch, queue.qsize(), inputs.shape[0], inputs.shape[1], inputs.shape[2]))

        # no data from queue
        if inputs.shape[0] == 0:
            # close one loader
            train_loader_count -= 1
            logger.debug('left train_loader: %d' % (train_loader_count))

            # if every loader closed, stop training
            if train_loader_count == 0:
                break
            else:
                continue

        # flush gradients
        optimizer.zero_grad()

        # load tensors to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # output tensor shape: (batch_size, n_mfcc, n_frames)
        # forward pass
        output = model(inputs).to(device)

        # compute loss
        loss = criterion(output.contiguous().view(-1), targets.contiguous().view(-1))
        print_loss += loss.item()
        total_loss += loss.item()

        # backward pass
        loss.backward()
        optimizer.step()

        # log progress
        if batch % print_batch == 0:
            # compute elapsed time
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            # log
            log_str = 'batch: {:4d}/{:4d}, batch size: {:3d} loss: {:.4f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'. \
                format(batch, total_batch_size, batch_size, print_loss / print_batch, elapsed, epoch_elapsed,
                       train_elapsed)
            logger.info(log_str)

            # reset
            begin = time.time()
            print_loss = 0

    # finish logging
    logger.info('train() completed')

    return total_loss / batch


def evaluate(model, dataloader, queue, criterion, device):
    total_loss = 0.
    batch = 0

    # set model to eval mode
    model.eval()

    # begin logging
    logger.info('evaluate() start')
    begin = time.time()

    with torch.no_grad():
        while True:
            batch += 1

            if queue.empty():
                logger.debug('queue is empty')

            # input, target tensor shapes: (batch_size, n_mfcc, n_frames)
            inputs, targets = queue.get()
            batch_size = inputs.shape[0]

            # if no data from queue, end evaluation
            if batch_size == 0:
                break

            # load tensors to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # output tensor shape: (batch_size, n_mfcc, n_frames)
            # forward pass
            output = model(inputs).to(device)

            # compute loss
            loss = criterion(output.contiguous().view(-1), targets.contiguous().view(-1))
            total_loss += loss.item()

    # finish logging
    logger.info('evaluate() completed')

    return total_loss / batch


def load(filename, model, optimizer):
    state = torch.load(filename)
    model.load_state_dict(state['model'])
    if 'optimizer' in state and optimizer:
        optimizer.load_state_dict(state['optimizer'])
    logger.info('checkpoint loaded from ' + filename)
    return state['best_loss']


def save(filename, model, optimizer, best_loss):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': best_loss
    }
    torch.save(state, filename)
    logger.info('checkpoint saved at ' + filename)


def import_paths():
    logger.info("loading file paths")
    f = open(FILE_PATHS, "r")
    lines = f.readlines()
    for line in lines:
        file_paths.append(line[:-1])
    f.close()


def split_dataset(batch_size, valid_ratio, num_workers):
    # split train/val dataset
    # construct BaseDataset objects using file_paths
    train_loader_count = num_workers
    file_num = len(file_paths)
    batch_num = math.ceil(file_num / batch_size)

    # train_batch_num: number of batches in training data
    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num

    batch_num_per_train_loader = math.ceil(train_batch_num/num_workers)

    train_begin = 0
    train_end_raw_id = 0
    train_dataset_list = list()

    # train_dataset_list: list of BaseDataset objects of training data
    for i in range(num_workers):
        train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

        train_begin_raw_id = train_begin * batch_size
        train_end_raw_id = train_end * batch_size

        train_dataset_list.append(
            BaseDataset(file_paths[train_begin_raw_id:train_end_raw_id], train_mode=True))

        train_begin = train_end

    # val_dataset: BaseDataset object of validation data
    valid_dataset = BaseDataset(file_paths[train_end_raw_id:], train_mode=False)

    logger.debug('train_dataset_list contains %d elements' % len(train_dataset_list))

    return train_batch_num, train_dataset_list, valid_dataset


def main():
    # set random seed
    """
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    """

    # set device
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    # initialize model
    model = AutoEncoder(d=n_mfcc*n_frames)
    logger.info("model: " + str(model))

    # load model to device
    model = nn.DataParallel(model).to(device)

    # set optimizer and loss
    optimizer = optim.Adam(model.module.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='mean').to(device)

    # load pre-trained model
    try:
        logger.info('loading checkpoint...')
        loaded_loss = load(SAVE_PATH + "best_eval.pt", model, optimizer)
    except (RuntimeError, FileNotFoundError):
        logger.info('no checkpoint loaded')
    else:
        logger.info('checkpoint loaded; current best loss %0.4f' % loaded_loss)

    # import data file paths
    import_paths()
    for i in range(10):
        logger.debug(file_paths[i])
    logger.debug("...")
    # randomly shuffle data
    random.shuffle(file_paths)

    best_loss = 1e20
    begin_epoch = 0

    # split training and validation data
    train_batch_num, train_dataset_list, valid_dataset = split_dataset(batch_size=batch_size, valid_ratio=valid_ratio, num_workers=num_workers)
    logger.debug('number of batches: %d' % train_batch_num)

    # begin logging
    logger.info('start')

    train_begin = time.time()
    for epoch in range(begin_epoch, max_epochs):
        logger.info("epoch: %d" % (epoch+1))

        train_queue = queue.Queue(num_workers * 2)
        train_loader = MultiLoader(train_dataset_list, train_queue, batch_size, num_workers)

        train_loader.start()
        train_loss = train(model, train_batch_num, train_queue, criterion, optimizer, device, train_begin, num_workers, print_batch=5)
        logger.info("Epoch %d Training Loss %0.4f" % (epoch, train_loss))
        train_loader.join()

        valid_queue = queue.Queue(num_workers * 2)
        valid_loader = BaseDataLoader(valid_dataset, valid_queue, batch_size, 0)

        valid_loader.start()
        logger.info("start eval")
        eval_loss = evaluate(model, valid_loader, valid_queue, criterion, device)
        valid_loader.join()
        logger.info("end eval")

        # save every epoch
        save_name = SAVE_PATH + "model_%03d" % epoch + ".pt"
        save(save_name, model, optimizer, train_loss)

        # save best loss model
        is_best_loss = (eval_loss < best_loss)
        if is_best_loss:
            logger.info("best model: " + save_name)
            best_loss = eval_loss
            save_name = SAVE_PATH + "best_eval.pt"
            save(save_name, model, optimizer, best_loss)

    return 0


if __name__ == "__main__":
    main()

