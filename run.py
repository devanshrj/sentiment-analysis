"""
Train a model on IMDb dataset.
"""

import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from json import dumps
from torchtext import datasets
from transformers import BertModel

from args import get_train_args
from preprocess import Dataset
from model import BERTSentiment
from utils import *


def main(args):
    # set up logs and device
    args.save_dir = get_save_dir(args.save_dir, args.name)
    log = get_logger(args.save_dir, args.name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    # set random seed
    log.info(f'Using random seed {args.seed}...')
    set_seeds(args.seed)

    # create dataset using torchtext
    log.info(f'Build data fields and {args.bert_variant} tokenizer...')
    dataset = Dataset(args.bert_variant)
    TEXT, LABEL = dataset.get_fields()

    # train:valid:test = 17500:7500:25000
    log.info('Build IMDb dataset using torchtext.datasets...')
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split(
        random_state=random.seed(args.seed))

    # iterators
    train_iterator, valid_iterator, test_iterator = dataset.get_iterators(
        train_data, valid_data, test_data, args.batch_size, device)

    # build LABEL vocabulary
    LABEL.build_vocab(train_data)

    # define model
    log.info('Building model...')
    model = BERTSentiment(args.bert_variant, args.hidden_dim, args.output_dim,
                          args.n_layers, args.bidirectional, args.dropout)

    # optimizer
    optimizer = optim.Adam(model.parameters())

    # criterion
    criterion = nn.BCEWithLogitsLoss()

    # place model and criterion on device
    model = model.to(device)
    criterion = criterion.to(device)

    # train set and validation set
    best_valid_loss = float('inf')
    for epoch in range(args.num_epochs):

        start_time = time.time()

        log.info(f'Training, epoch = {epoch}...')
        train_loss, train_acc = train(model, train_iterator, optimizer,
                                      criterion)

        log.info(f'Evaluating, epoch = {epoch}...')
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            log.info(f'Saving best model...')
            best_valid_loss = valid_loss
            torch.save(model.state_dict(),
                       f'{args.save_dir}/sent-bert-model.pt')

    log.info('Model trained and evaluated...')

    # test set
    log.info('Testing...')
    model.load_state_dict(torch.load(f'{args.save_dir}/sent-bert-model.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


def train(model, iterator, optimizer, criterion):
    """
    Train function.
    Params:
        model: instance of model
        iterator: iterator for train set
        optimizer
        criterion: loss function
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    with tqdm.tqdm(total=len(iterator)) as progress_bar:
        for batch in iterator:
            optimizer.zero_grad()

            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    """
    Evaluate function.
    Params:
        model: instance of model
        iterator: iterator for train set
        criterion: loss function
    """
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad(), tqdm.tqdm(total=len(iterator)) as progress_bar:
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == '__main__':
    main(get_train_args())
