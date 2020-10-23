"""
Predict the sentiment of an input sentence using the trained model.
"""

import torch

from args import get_predict_args
from model import BERTSentiment
from preprocess import Dataset
from utils import *

# set up args and device
args = get_predict_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# tokenizer from class Dataset
dataset = Dataset(args.bert_variant)
tokenizer = dataset.get_tokenizer()

# build model from saved dict
model = BERTSentiment(args.bert_variant, args.hidden_dim, args.output_dim,
                      args.n_layers, args.bidirectional, args.dropout)
model.load_state_dict(torch.load(
    f'{args.save_dir}/{args.model_name}', map_location=device))


def predict_sentiment(sentence):
    """
    Predict sentiment of a sentence input by the user.
    Params:
        model: trained model
        tokenizer: BertTokenizer to tokenize the input sentence
        sentence (str): sentence input by user to predict sentiment
    Returns:
        sentiment (str): sentiment predicted by the model for the input sentence
        sentiment_prob (float): probability of the predicted sentiment
    """
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:dataset.max_input_len-2]
    indexed = [dataset.init_token_idx] + \
        tokenizer.convert_tokens_to_ids(tokens) + [dataset.eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    sentiment_prob = prediction.item()
    sentiment = 'Positive' if sentiment_prob >= 0.5 else 'Negative'
    return sentiment, sentiment_prob
