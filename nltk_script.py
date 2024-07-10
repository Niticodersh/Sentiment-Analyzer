import string
import plotly.express as px

import nltk
from nltk import download
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Function to download NLTK resources if not already downloaded
def download_nltk_resources():
    try:
        sid = SentimentIntensityAnalyzer()
    except LookupError:
        download('vader_lexicon')
        download('punkt')

# Check if NLTK resources are downloaded
download_nltk_resources()

text = input("Enter the content to be analyzed")
lowercase = text.lower()
cleaned_txt = lowercase.translate(str.maketrans('','',string.punctuation))
labels = ['Negative', 'Neutral', 'Positive']
def sentiment_analyze(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    return list(score.values())[:-1]

scores = sentiment_analyze(cleaned_txt)
print(scores)
fig = px.pie(values=scores, names=labels, title="Sentiment Distribution")
fig.show()


