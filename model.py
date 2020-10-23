"""
Model class.
"""

import torch
import torch.nn as nn

from transformers import BertModel


class BERTSentiment(nn.Module):
    def __init__(self, bert_variant, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        """
        Model: Embedding (BERT) -> Multilayer GRU -> Linear
        Params:
            bert: BertModel from huggingface transformers
            hidden_dim: dimensions of hidden layers
            output_dim: dimensions of output layer
            n_layers: number of layers for GRU
            bidirectional (boolean): if true, GRU is bidirectional
            dropout: dropout probability
        """
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_variant)
        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch size, sent len]
        # embedded = [batch size, sent len, emb dim]
        with torch.no_grad():
            embedded = self.bert(text)[0]

        # hidden = [n layers * n directions, batch size, emb dim]
        _, hidden = self.rnn(embedded)

        # hidden = [batch size, hid dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(
                torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # output = [batch size, out dim]
        output = self.out(hidden)

        return output
