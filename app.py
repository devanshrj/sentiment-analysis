from flask import Flask, request, render_template

from args import get_app_args()
from model import BERTSentiment
from preprocess import Dataset
from utils import *

# get arguments
get_app_args()

# tokenizer from class Dataset
dataset = Dataset(args.bert_variant)
tokenizer = dataset.get_tokenizer()

# build model from saved dict
model = BERTSentiment(args.bert_variant, args.hidden_dim, args.output_dim,
                      args.n_layers, args.bidirectional, args.dropout)
model.load_state_dict(torch.load(
    f'{args.save_dir}/{args.name}/bert-sent-model.pt'))


def predict_sentiment(model, tokenizer, sentence):
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
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + \
        tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    sentiment_prob = prediction.item()
    sentiment = 'positive' if sentiment_prob >= 0.5 else 'negative'
    return sentiment, sentiment_prob


# flask app
app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('text-form.html')


#  @app.route('/submit', methods=['POST'])
#  def submit():
#      return 'You entered: {}'.format(request.form['text'])

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form['text']
        sentiment, sentiment_prob = predict_sentiment(
            model, tokenizer, sentence)
        return f'Predicted Sentiment: {sentiment} \n Sentiment Probability: {sentiment_prob}''


if __name__ == '__main__':
    app.run()
