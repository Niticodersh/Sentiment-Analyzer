import requests
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AdamW, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
import math
import plotly.express as px


import os
import gdown

model_path = "best_model.ckpt"
model_url = "https://drive.google.com/uc?id=1-zQyH3AI9MgvicfVqJhnqs875tQi5MFO"

def download_model(url, dest):
    if not os.path.exists(dest):
        gdown.download(url, dest, quiet=False)
        print(f"Model downloaded to {dest}")
    else:
        print(f"Model already exists at {dest}")

download_model(model_url, model_path)

# Pre-process input text
txt = input("Enter the content to be analyzed: ")
txt_words = ["@user" if word.startswith('@') and len(word) > 1 else "http" if word.startswith('http') else word for word in txt.split()]
txt_proc = " ".join(txt_words)

# Model architecture
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
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
labels = ['Negative', 'Neutral', 'Positive']
mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

tokens = tokenizer.encode_plus(
    txt_proc,
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

# Print the results
print(f"The predicted sentiment is {mapping[max_index.item()]}, with a probability of {float(max_prob.item()):.4f}")
fig = px.pie(values=scores, names=labels, title="Sentiment Distribution")
fig.show()
