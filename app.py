import requests
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AdamW, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
import math
import plotly.express as px
import gdown
from scipy.special import softmax
import streamlit as st
import string
import nltk
import time
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
# Set up the Streamlit page
st.set_page_config(page_title="Penguin Interprets", page_icon=":penguin:", layout="wide")
st.title(":penguin: Sentiment Analyzer")
msg = "Enter the content to be analyzed"
txt = st.text_input(label=msg, value="")
option = st.selectbox("Select a analyzer",("roBERTa Analyzer", "NLTK Analyzer", "Fine-tuned roBERTa (for financial sentiments) warning: Model size is 1.4 GB, use only if you have the required space", "Compare Analyzers"))
print("Option chosen", option)
analyze_button = st.button('Analyze')

def fine_tuned_roBERTa(text):
    model_path = "model.ckpt"
    # model_url = "https://drive.google.com/file/d/1CuIwhkqWu1_M_rjoHDAmFB2X5IL2UCf9"
    model_url = "https://drive.google.com/uc?id=1CuIwhkqWu1_M_rjoHDAmFB2X5IL2UCf9"


    # def download_model(url, dest):
    #     if not os.path.exists(dest):
    #         with st.spinner('This is one-time process, once model is downloaded, you can use it as many times. Downloading fine-tuned roBERTa...'):
    #             response = gdown.download(url, dest, quiet=False)
    #         st.success('Model downloaded successfully!')

    def download_model(url, dest):
        if not os.path.exists(dest):
            retries = 5
            for attempt in range(retries):
                try:
                    with st.spinner(
                            'This is a one-time process, once the model is downloaded, you can use it as many times. Downloading fine-tuned roBERTa...'):
                        gdown.download(url, dest, quiet=False)
                    st.success('Model downloaded successfully!')
                    break
                except Exception as e:
                    st.warning(f"Attempt {attempt + 1} of {retries} failed: {e}")
                    time.sleep(5)  # Wait before retrying
                    if attempt == retries - 1:
                        st.error("Failed to download model after several attempts.")
                        raise e



    download_model(model_url, model_path)

    class SentimentClassifier(pl.LightningModule):
        def __init__(self, config: dict):
            super().__init__()
            self.config = config
            self.pretrained_model = AutoModel.from_pretrained(self.config['model_name'], return_dict=True)
            self.hidden = nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
            self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, self.config['num_classes'])
            torch.nn.init.xavier_uniform_(self.hidden.weight)
            torch.nn.init.xavier_uniform_(self.classifier.weight)
            self.loss_fun = nn.CrossEntropyLoss(reduction='mean')
            self.dropout = nn.Dropout(self.config['dropout'])
            self.relu = nn.ReLU()

        def forward(self, input_ids, attention_mask, labels=None):
            output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = torch.mean(output.last_hidden_state, dim=1)
            pooled_output = self.hidden(pooled_output)
            pooled_output = self.dropout(pooled_output)
            pooled_output = self.relu(pooled_output)
            logits = self.classifier(pooled_output)
            loss = 0
            if labels is not None:
                loss = self.loss_fun(logits, labels)
            return logits, loss

    # Config
    config = {
        'model_name': 'roberta-base',
        'num_classes': 3,
        'batch_size': 32,
        'lr': 1.5e-6,
        'warmup_ratio': 0.2,
        'w_decay': 0.001,
        'n_epochs': 30,
        'dropout': 0.2,
    }

    # Load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentimentClassifier.load_from_checkpoint(model_path, config=config).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    labels = ['Negative', 'Neutral', 'Positive']
    mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True
    )

    input_ids = tokens['input_ids'].flatten().to(device)
    attention_mask = tokens['attention_mask'].flatten().to(device)
    logits, _ = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))

    # Apply softmax to convert logits to probabilities
    probs = torch.softmax(logits, dim=1)
    max_prob, max_index = torch.max(probs, dim=1)

    scores = probs.tolist()[0]
    return scores, labels, max_prob, max_index, mapping

def find_max_element_and_index(nums):
    print("nums",nums)

    max_value = nums[0]  # Initialize max_value with the first element
    max_index = 0         # Initialize max_index with 0

    # Iterate through the list to find the maximum value and its index
    for i in range(1, len(nums)):
        if nums[i] > max_value:
            max_value = nums[i]
            max_index = i

    return max_value, max_index

def NLTK_Analysis(text):
    labels = ['Negative', 'Neutral', 'Positive']
    mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    lowercase = text.lower()
    cleaned_txt = lowercase.translate(str.maketrans('', '', string.punctuation))
    score = SentimentIntensityAnalyzer().polarity_scores(cleaned_txt)
    scores = list(score.values())[:-1]
    max_prob, max_index = find_max_element_and_index(scores)

    return scores, labels, max_prob, max_index, mapping
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
    mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

    # Sentiment analysis
    encoded_txts = tokenizer(txt_proc, return_tensors='pt')
    output = model(**encoded_txts)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    max_prob, max_index = find_max_element_and_index(scores)
    return scores, labels, max_prob, max_index, mapping


if analyze_button and txt:
    print(txt)  # For debugging in the terminal

    if option == "roBERTa Analyzer":
        # Analyze sentiment
        scores, labels, max_prob, max_index, mapping = roBERTa_Analysis(txt)
        print(scores)  # For debugging in the terminal
        st.write(f"The predicted sentiment is {mapping[max_index]}, with a probability of {float(max_prob):.4f}")

        # Plot the results
        fig = px.pie(values=scores, names=labels, title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if option == "NLTK Analyzer":
        # Analyze sentiment
        scores, labels, max_prob, max_index, mapping= NLTK_Analysis(txt)
        print(scores)  # For debugging in the terminal
        st.write(f"The predicted sentiment is {mapping[max_index]}, with a probability of {float(max_prob):.4f}")

        # Plot the results
        fig = px.pie(values=scores, names=labels, title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if option == "Fine-tuned roBERTa (for financial sentiments) warning: Model size is 1.4 GB, use only if you have the required space":
        scores, labels, max_prob, max_index, mapping = fine_tuned_roBERTa(txt)
        print(scores)
        st.write(f"The predicted sentiment is {mapping[max_index.item()]}, with a probability of {float(max_prob.item()):.4f}")
        # Plot the results
        fig = px.pie(values=scores, names=labels, title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if option == "Compare Analyzers":
        # Analyze sentiment
        roBERTa_scores, roBERTa_labels,_ ,_ , _  = roBERTa_Analysis(txt)
        print(roBERTa_scores)  # For debugging in the terminal

        NLTK_scores, NLTK_labels,_ ,_ , _  = NLTK_Analysis(txt)
        print(NLTK_scores)  # For debugging in the terminal

        fine_tuned_roBERTa_scores, fine_tuned_roBERTa_labels,_ ,_ , _ = fine_tuned_roBERTa(txt)
        print(fine_tuned_roBERTa_scores)  # For debugging in the terminal



        col1, col2, col3 = st.columns(3)

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

        with col3:
            # Plot the results for fine-tuned roBERTa
            fig_roBERTa = px.pie(values=fine_tuned_roBERTa_scores, names=fine_tuned_roBERTa_labels, title="Fine Tuned roBERTa Sentiment Distribution")
            st.subheader("Fine-tuned roBERTa Analyzer")
            st.plotly_chart(fig_roBERTa, use_container_width=True)

