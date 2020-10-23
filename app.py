from flask import Flask, request, render_template
from predict import predict_sentiment

# setup app
app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('text-form.html')


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form['text']
        sentiment, sentiment_prob = predict_sentiment(sentence)
        return f'Predicted Sentiment: {sentiment} \n Sentiment Probability: {sentiment_prob}'


if __name__ == '__main__':
    app.run()
