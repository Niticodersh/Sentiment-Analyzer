from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import plotly.express as px

#----------------------------------------------------------

txt = input("Enter the content to be analyzed")

#pre-process txt
txt_words = []
for word in txt.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'
    elif word.startswith('http'):
        word = 'http'
    txt_words.append(word)

txt_proc = " ".join(txt_words)

#load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['Negative', 'Neutral', 'Positive']

#sentiment analysis
encoded_txts = tokenizer(txt_proc, return_tensors='pt')
output = model(**encoded_txts)

scores = output[0][0].detach().numpy()
scores = softmax(scores)
print(scores)
fig = px.pie(values=scores, names=labels, title="Sentiment Distribution")
fig.show()