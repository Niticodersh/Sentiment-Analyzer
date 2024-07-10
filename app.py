from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import streamlit as st
import plotly.express as px
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Set up the Streamlit page
st.set_page_config(page_title="Penguin Interprets", page_icon=":penguin:", layout="wide")
st.title(":penguin: Sentiment Analyzer")
msg = "Enter the content to be analyzed"
txt = st.text_input(label=msg, value="")
option = st.selectbox("Select a analyzer",("roBERTa Analyzer", "NLTK Analyzer", "Compare Analyzers"))
print("Option chosen", option)
analyze_button = st.button('Analyze')

def NLTK_Analysis(text):
    labels = ['Negative', 'Neutral', 'Positive']
    lowercase = text.lower()
    cleaned_txt = lowercase.translate(str.maketrans('', '', string.punctuation))
    score = SentimentIntensityAnalyzer().polarity_scores(cleaned_txt)
    scores = list(score.values())[:-1]
    return scores, labels
# Define the analysis function
def roBERTa_Analysis(text):
    # Pre-process text
    txt_words = []
    for word in text.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = 'http'
        txt_words.append(word)

    txt_proc = " ".join(txt_words)

    # Load model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    labels = ['Negative', 'Neutral', 'Positive']

    # Sentiment analysis
    encoded_txts = tokenizer(txt_proc, return_tensors='pt')
    output = model(**encoded_txts)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores, labels


if analyze_button and txt:
    print(txt)  # For debugging in the terminal

    if option == "roBERTa Analyzer":
        # Analyze sentiment
        scores, labels = roBERTa_Analysis(txt)
        print(scores)  # For debugging in the terminal

        # Plot the results
        fig = px.pie(values=scores, names=labels, title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if option == "NLTK Analyzer":
        # Analyze sentiment
        scores, labels = NLTK_Analysis(txt)
        print(scores)  # For debugging in the terminal

        # Plot the results
        fig = px.pie(values=scores, names=labels, title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if option == "Compare Analyzers":
        # Analyze sentiment
        roBERTa_scores, roBERTa_labels = roBERTa_Analysis(txt)
        print(roBERTa_scores)  # For debugging in the terminal

        NLTK_scores, NLTK_labels = NLTK_Analysis(txt)
        print(NLTK_scores)  # For debugging in the terminal

        col1, col2 = st.columns(2)

        with col1:
            # Plot the results for roBERTa
            fig_roBERTa = px.pie(values=roBERTa_scores, names=roBERTa_labels, title="roBERTa Sentiment Distribution")
            st.subheader("roBERTa Analyzer")
            st.plotly_chart(fig_roBERTa, use_container_width=True)

        with col2:
            # Plot the results for NLTK
            fig_NLTK = px.pie(values=NLTK_scores, names=NLTK_labels, title="NLTK Sentiment Distribution")
            st.subheader("NLTK Analyzer")
            st.plotly_chart(fig_NLTK, use_container_width=True)

