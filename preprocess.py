"""
Preprocess class.
"""

import torch

from torchtext import data
from transformers import BertTokenizer


class Dataset():
    """
    """

    def __init__(self, bert_variant):
        """
        Create tokenizer, special tokens, max length for tokenizer.
        Params:
            bert_variant (string): Model class for BertTokenizer
        """
        self.tokenizer = BertTokenizer.from_pretrained(bert_variant)

        # special tokens
        self.init_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token

        # indexes for special tokens
        self.init_token_idx = self.tokenizer.cls_token_id
        self.eos_token_idx = self.tokenizer.sep_token_id
        self.pad_token_idx = self.tokenizer.pad_token_id
        self.unk_token_idx = self.tokenizer.unk_token_id

        # maximum input length for BertModel from the tokenizer
        self.max_input_len = self.tokenizer.max_model_input_sizes[bert_variant]

    def get_tokenizer(self):
        """
        Returns:
            tokenizer
        """
        return self.tokenizer

    def get_fields(self):
        """
        Create TEXT and LABEL fields.
        Params:
        Returns:
            TEXT (torchtext.data.Field): Data Field for text
            LABEL (torchtext.data.LabelField): Label Field for labels
        """

        def tokenize_and_cut(sentence):
            """ 
            Custom tokenization function.
            Tokenizes sentence and reduces to max length expected by the BertModel
            Params:
                sentence (str): input sentence to tokenize
            Returns:
                tokens (list[int]): list of integers corresponding to words in sentence
            """
            tokens = self.tokenizer.tokenize(sentence)
            tokens = tokens[:self.max_input_len - 2]
            return tokens

        TEXT = data.Field(batch_first=True,
                          use_vocab=False,
                          tokenize=tokenize_and_cut,
                          preprocessing=self.tokenizer.convert_tokens_to_ids,
                          init_token=self.init_token_idx,
                          eos_token=self.eos_token_idx,
                          pad_token=self.pad_token_idx,
                          unk_token=self.unk_token_idx)

        LABEL = data.LabelField(dtype=torch.float)
        return TEXT, LABEL

    def get_iterators(self, train_data, valid_data, test_data, batch_size, device):
        """
        Create iterator for dataset.
        Params:
            train_data: train set
            valid_data: validation set
            test_data: test set
            batch_size (int): batch size
            device (str): device (cuda or cpu)
        Returns:
            train_iterator: iterator for train set 
            valid_iterator: iterator for valid set 
            test_iterator: iterator for test set 
        """
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=batch_size,
            device=device
        )
        return train_iterator, valid_iterator, test_iterator
