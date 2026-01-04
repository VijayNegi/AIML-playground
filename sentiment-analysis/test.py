import torch
import transformers

transformers.__version__

from transformers import pipeline 
sentiment_pipeline = pipeline("sentiment-analysis") 
data = ["I love you", "I hate you"]
sentiment_pipeline(data)
specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis") 
specific_model(data)
specific_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment") 
results = specific_model(data)
print(results)
from datasets import load_dataset
imdb = load_dataset("imdb")
small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])