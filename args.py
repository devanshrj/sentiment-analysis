"""
Command line arguments for train.py.
"""

import argparse


def get_train_args():
    """
    Arguments needed in train.py.
    """
    parser = argparse.ArgumentParser('Train a model on IMDb Dataset')
    add_train_test_args(parser)

    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        help='Random seed for reproducibility.')

    parser.add_argument('--num_epochs',
                        type=int,
                        default=5,
                        help='Number of epochs for which to train')

    parser.add_argument('--output_dim',
                        type=int,
                        default=1,
                        help='Output dimensions')

    parser.add_argument('--num_layers',
                        type=int,
                        default=2,
                        help='Number of GRU layers')

    parser.add_argument('--bidirectional',
                        type=bool,
                        default=True,
                        help='Directionality of GRU')

    parser.add_argument('--dropout',
                        type=float,
                        default=0.25,
                        help='Dropout probability')


def add_train_test_args(parser):
    """
    Arguments needed in both train.py and test.py.
    Params:
        parser
    """
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify train or test run')

    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Batch size')

    parser.add_argument('--hidden_dim',
                        type=int,
                        default=256
                        help='Hidden dimension')

    parser.add_argument('--bert_variant',
                        type=str,
                        default='bert-base-uncased',
                        help='Variant of BERT Model from transformers')