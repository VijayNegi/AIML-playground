#pip install -q transformers
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
data = ["I love you", "I hate you"]
result= sentiment_pipeline(data)
print(result)

specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
specific_result = specific_model(data)
print(specific_result)