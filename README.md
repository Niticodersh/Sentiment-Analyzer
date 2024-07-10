# 🐧 Welcome to Penguin Interprets 
This github repo provides the code to analyze the sentiments as positive, negative and neutral through a pie chart visualization. 

To use this repository: 
First clone it: `git clone https://github.com/Niticodersh/Sentiment-Analyzer.git` 

Then install dependencies: `pip install -r requirements.txt` 

This code provides sentiment analysis using two analyzers.
1. **roBERTa Analyzer**: It uses RoBERTa model (Robustly Optimized BERT Pretraining Approach) by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. It is based on Google’s BERT model released in 2018.
2. **NLTK Analyzer**: It uses nltk inbuilt library SentimentIntensityAnalyzer. 
   
Run the CLI based python files to see the model results. 

For *roBERTa Analyzer* : `python roBERTa_script.py` 

For *NLTK Analyzer* : `python nltk_script.py` 

From streamlit deployment run *app.py* : `streamlit run app.py` 

We have our app deployed on streamlit to directly use it. Go to this streamlit webApp: https://penguin-interprets-sentiment-analyzer-niticodersh.streamlit.app/ 
