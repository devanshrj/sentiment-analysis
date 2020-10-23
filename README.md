## Sentiment Analysis

Objective: To build a model for Sentiment Analysis on the IMDb dataset using TorchText. <br>

Dataset: [IMDb](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
<br>

### Model

- Architecture
  - BERT Base as Embedding Layer
  - Multilayer GRU
  - Linear Layer
- Model Accuracy - Test Loss: 0.198 | Test Acc: 92.10%
  <br>

### Instructions to Run

1. Clone the repository <br>
   `git clone `
2. Install the requirements <br>
   `pip install -r requirements.txt`
3. Local deployment <br>
   `python app.py` or `FLASK_ENV=development FLASK_APP=app.py flask run`
   <br>

### Learning Outcomes

- TorchText
- Model Deployment with Flask
  <br>

### To Do

- [] Improve interface
  <br>

### Resources

- [Deployment](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
-
