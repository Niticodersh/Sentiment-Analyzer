import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
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


