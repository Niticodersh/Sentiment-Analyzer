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
import sys
from nltk import download
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#-----------gdown download.py used explicitly due to stderr issue on deployment
import email.utils
import os
import os.path as osp
import re
import shutil
import sys
import tempfile
import textwrap
import time
import urllib.parse
from http.cookiejar import MozillaCookieJar

import bs4
import requests
import tqdm

# from ._indent import indent
# from .exceptions import FileURLRetrievalError
# from .parse_url import parse_url

CHUNK_SIZE = 512 * 1024  # 512KB
home = osp.expanduser("~")

# textwrap.indent for Python2
def indent(text, prefix):
    def prefixed_lines():
        for line in text.splitlines(True):
            yield (prefix + line if line.strip() else line)

    return "".join(prefixed_lines())

class FileURLRetrievalError(Exception):
    pass


class FolderContentsMaximumLimitError(Exception):
    pass

import re
import urllib.parse
import warnings


def is_google_drive_url(url):
    parsed = urllib.parse.urlparse(url)
    return parsed.hostname in ["drive.google.com", "docs.google.com"]


def parse_url(url, warning=True):
    """Parse URLs especially for Google Drive links.

    file_id: ID of file on Google Drive.
    is_download_link: Flag if it is download link of Google Drive.
    """
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    is_gdrive = is_google_drive_url(url=url)
    is_download_link = parsed.path.endswith("/uc")

    if not is_gdrive:
        return is_gdrive, is_download_link

    file_id = None
    if "id" in query:
        file_ids = query["id"]
        if len(file_ids) == 1:
            file_id = file_ids[0]
    else:
        patterns = [
            r"^/file/d/(.*?)/(edit|view)$",
            r"^/file/u/[0-9]+/d/(.*?)/(edit|view)$",
            r"^/document/d/(.*?)/(edit|htmlview|view)$",
            r"^/document/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$",
            r"^/presentation/d/(.*?)/(edit|htmlview|view)$",
            r"^/presentation/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$",
            r"^/spreadsheets/d/(.*?)/(edit|htmlview|view)$",
            r"^/spreadsheets/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, parsed.path)
            if match:
                file_id = match.groups()[0]
                break

    if warning and not is_download_link:
        warnings.warn(
            "You specified a Google Drive link that is not the correct link "
            "to download a file. You might want to try `--fuzzy` option "
            "or the following url: {url}".format(
                url="https://drive.google.com/uc?id={}".format(file_id)
            )
        )

    return file_id, is_download_link

def get_url_from_gdrive_confirmation(contents):
    url = ""
    for line in contents.splitlines():
        m = re.search(r'href="(\/uc\?export=download[^"]+)', line)
        if m:
            url = "https://docs.google.com" + m.groups()[0]
            url = url.replace("&amp;", "&")
            break
        soup = bs4.BeautifulSoup(line, features="html.parser")
        form = soup.select_one("#download-form")
        if form is not None:
            url = form["action"].replace("&amp;", "&")
            url_components = urllib.parse.urlsplit(url)
            query_params = urllib.parse.parse_qs(url_components.query)
            for param in form.findChildren("input", attrs={"type": "hidden"}):
                query_params[param["name"]] = param["value"]
            query = urllib.parse.urlencode(query_params, doseq=True)
            url = urllib.parse.urlunsplit(url_components._replace(query=query))
            break
        m = re.search('"downloadUrl":"([^"]+)', line)
        if m:
            url = m.groups()[0]
            url = url.replace("\\u003d", "=")
            url = url.replace("\\u0026", "&")
            break
        m = re.search('<p class="uc-error-subcaption">(.*)</p>', line)
        if m:
            error = m.groups()[0]
            raise FileURLRetrievalError(error)
    if not url:
        raise FileURLRetrievalError(
            "Cannot retrieve the public link of the file. "
            "You may need to change the permission to "
            "'Anyone with the link', or have had many accesses. "
            "Check FAQ in https://github.com/wkentaro/gdown?tab=readme-ov-file#faq.",
        )
    return url


def _get_filename_from_response(response):
    content_disposition = urllib.parse.unquote(response.headers["Content-Disposition"])

    m = re.search(r"filename\*=UTF-8''(.*)", content_disposition)
    if m:
        filename = m.groups()[0]
        return filename.replace(osp.sep, "_")

    m = re.search('attachment; filename="(.*?)"', content_disposition)
    if m:
        filename = m.groups()[0]
        return filename

    return None


def _get_modified_time_from_response(response):
    if "Last-Modified" not in response.headers:
        return None

    raw = response.headers["Last-Modified"]
    if raw is None:
        return None

    return email.utils.parsedate_to_datetime(raw)


def _get_session(proxy, use_cookies, user_agent, return_cookies_file=False):
    sess = requests.session()

    sess.headers.update({"User-Agent": user_agent})

    if proxy is not None:
        sess.proxies = {"http": proxy, "https": proxy}
        print("Using proxy:", proxy)

    # Load cookies if exists
    cookies_file = osp.join(home, ".cache/gdown/cookies.txt")
    if use_cookies and osp.exists(cookies_file):
        cookie_jar = MozillaCookieJar(cookies_file)
        cookie_jar.load()
        sess.cookies.update(cookie_jar)

    if return_cookies_file:
        return sess, cookies_file
    else:
        return sess


def gdown_download(
    url=None,
    output=None,
    quiet=False,
    proxy=None,
    speed=None,
    use_cookies=True,
    verify=True,
    id=None,
    fuzzy=False,
    resume=False,
    format=None,
    user_agent=None,
    log_messages=None,
):

    if not (id is None) ^ (url is None):
        raise ValueError("Either url or id has to be specified")
    if id is not None:
        url = "https://drive.google.com/uc?id={id}".format(id=id)
    if user_agent is None:
        # We need to use different user agent for file download c.f., folder
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"  # NOQA: E501
    if log_messages is None:
        log_messages = {}

    url_origin = url

    sess, cookies_file = _get_session(
        proxy=proxy,
        use_cookies=use_cookies,
        user_agent=user_agent,
        return_cookies_file=True,
    )

    gdrive_file_id, is_gdrive_download_link = parse_url(url, warning=not fuzzy)

    if fuzzy and gdrive_file_id:
        # overwrite the url with fuzzy match of a file id
        url = "https://drive.google.com/uc?id={id}".format(id=gdrive_file_id)
        url_origin = url
        is_gdrive_download_link = True

    while True:
        res = sess.get(url, stream=True, verify=verify)

        if not (gdrive_file_id and is_gdrive_download_link):
            break

        if url == url_origin and res.status_code == 500:
            # The file could be Google Docs or Spreadsheets.
            url = "https://drive.google.com/open?id={id}".format(id=gdrive_file_id)
            continue

        if res.headers["Content-Type"].startswith("text/html"):
            m = re.search("<title>(.+)</title>", res.text)
            if m and m.groups()[0].endswith(" - Google Docs"):
                url = (
                    "https://docs.google.com/document/d/{id}/export"
                    "?format={format}".format(
                        id=gdrive_file_id,
                        format="docx" if format is None else format,
                    )
                )
                continue
            elif m and m.groups()[0].endswith(" - Google Sheets"):
                url = (
                    "https://docs.google.com/spreadsheets/d/{id}/export"
                    "?format={format}".format(
                        id=gdrive_file_id,
                        format="xlsx" if format is None else format,
                    )
                )
                continue
            elif m and m.groups()[0].endswith(" - Google Slides"):
                url = (
                    "https://docs.google.com/presentation/d/{id}/export"
                    "?format={format}".format(
                        id=gdrive_file_id,
                        format="pptx" if format is None else format,
                    )
                )
                continue
        elif (
            "Content-Disposition" in res.headers
            and res.headers["Content-Disposition"].endswith("pptx")
            and format not in {None, "pptx"}
        ):
            url = (
                "https://docs.google.com/presentation/d/{id}/export"
                "?format={format}".format(
                    id=gdrive_file_id,
                    format="pptx" if format is None else format,
                )
            )
            continue

        if use_cookies:
            cookie_jar = MozillaCookieJar(cookies_file)
            for cookie in sess.cookies:
                cookie_jar.set_cookie(cookie)
            cookie_jar.save()

        if "Content-Disposition" in res.headers:
            # This is the file
            break

        # Need to redirect with confirmation
        try:
            url = get_url_from_gdrive_confirmation(res.text)
        except FileURLRetrievalError as e:
            message = (
                "Failed to retrieve file url:\n\n{}\n\n"
                "You may still be able to access the file from the browser:"
                "\n\n\t{}\n\n"
                "but Gdown can't. Please check connections and permissions."
            ).format(
                indent("\n".join(textwrap.wrap(str(e))), prefix="\t"),
                url_origin,
            )
            raise FileURLRetrievalError(message)

    filename_from_url = None
    last_modified_time = None
    if gdrive_file_id and is_gdrive_download_link:
        filename_from_url = _get_filename_from_response(response=res)
        last_modified_time = _get_modified_time_from_response(response=res)
    if filename_from_url is None:
        filename_from_url = osp.basename(url)

    if output is None:
        output = filename_from_url

    output_is_path = isinstance(output, str)
    if output_is_path and output.endswith(osp.sep):
        if not osp.exists(output):
            os.makedirs(output)
        output = osp.join(output, filename_from_url)

    if output_is_path:
        if resume and os.path.isfile(output):
            if not quiet:
                print(f"Skipping already downloaded file {output}")
            return output

        existing_tmp_files = []
        for file in os.listdir(osp.dirname(output) or "."):
            if file.startswith(osp.basename(output)) and file.endswith(".part"):
                existing_tmp_files.append(osp.join(osp.dirname(output), file))
        if resume and existing_tmp_files:
            if len(existing_tmp_files) != 1:
                print(
                    "There are multiple temporary files to resume:",
                )
                print("\n")
                for file in existing_tmp_files:
                    print("\t", file)
                print("\n")
                print(
                    "Please remove them except one to resume downloading.",
                )
                return
            tmp_file = existing_tmp_files[0]
        else:
            resume = False
            # mkstemp is preferred, but does not work on Windows
            # https://github.com/wkentaro/gdown/issues/153
            tmp_file = tempfile.mktemp(
                suffix=".part",
                prefix=osp.basename(output),
                dir=osp.dirname(output),
            )
        f = open(tmp_file, "ab")
    else:
        tmp_file = None
        f = output

    if tmp_file is not None and f.tell() != 0:
        start_size = f.tell()
        headers = {"Range": "bytes={}-".format(start_size)}
        res = sess.get(url, headers=headers, stream=True, verify=verify)
    else:
        start_size = 0

    if not quiet:
        print(log_messages.get("start", "Downloading...\n"), end="")
        if resume:
            print("Resume:", tmp_file)
        if url_origin != url:
            print("From (original):", url_origin)
            print("From (redirected):", url)
        else:
            print("From:", url)
        print(
            log_messages.get(
                "output", f"To: {osp.abspath(output) if output_is_path else output}\n"
            ),
            end="",
        )

    try:
        total = res.headers.get("Content-Length")
        if total is not None:
            total = int(total) + start_size
        if not quiet:
            pbar = tqdm.tqdm(total=total, unit="B", initial=start_size, unit_scale=True)
        t_start = time.time()
        for chunk in res.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            if not quiet:
                pbar.update(len(chunk))
            if speed is not None:
                elapsed_time_expected = 1.0 * pbar.n / speed
                elapsed_time = time.time() - t_start
                if elapsed_time < elapsed_time_expected:
                    time.sleep(elapsed_time_expected - elapsed_time)
        if not quiet:
            pbar.close()
        if tmp_file:
            f.close()
            shutil.move(tmp_file, output)
        if output_is_path and last_modified_time:
            mtime = last_modified_time.timestamp()
            os.utime(output, (mtime, mtime))
    finally:
        sess.close()

    return output

#------------------------------------------------------------------------------
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
# print("Option chosen", option)
analyze_button = st.button('Analyze')

def fine_tuned_roBERTa(text):
    model_path = "best_model.ckpt"
    # model_url = "https://drive.google.com/uc?id=1-zQyH3AI9MgvicfVqJhnqs875tQi5MFO"
    model_url = "https://drive.google.com/uc?id=1-zQyH3AI9MgvicfVqJhnqs875tQi5MFO&confirm=t&uuid=4a7d3dc2-cf40-48ba-bf55-574437c277ca"
    def download_model(url, dest):
        try:
            if not os.path.exists(dest):
                with st.spinner('Downloading fine-tuned roBERTa...'):
                    gdown_download(url, dest, quiet=False)
                st.success('Model downloaded successfully!')
            else:
                st.info('Model already exists, skipping download.')
        except Exception as e:
            st.error(f"Error downloading model: {e}")

    #     def download_model(url, dest):
#         if not os.path.exists(dest):
#             with st.spinner('This is one-time process, once model is downloaded, you can use it as many times. Downloading fine-tuned roBERTa...'):
#                 try:
#                     response = gdown.download(url, dest, quiet=False)
#                     st.success('Model downloaded successfully!')
#                 except BrokenPipeError:  # Added error handling
# #                     print("BrokenPipeError encountered during model download.", file=sys.stderr)
#                 except Exception as e:
# #                     print(f"An error occurred: {e}", file=sys.stderr)

    download_model(model_url, model_path)


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
#     print("nums",nums)

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
#     print(txt)  # For debugging in the terminal

    if option == "roBERTa Analyzer":
        # Analyze sentiment
        scores, labels, max_prob, max_index, mapping = roBERTa_Analysis(txt)
#         print(scores)  # For debugging in the terminal
        st.write(f"The predicted sentiment is {mapping[max_index]}, with a probability of {float(max_prob):.4f}")

        # Plot the results
        fig = px.pie(values=scores, names=labels, title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if option == "NLTK Analyzer":
        # Analyze sentiment
        scores, labels, max_prob, max_index, mapping= NLTK_Analysis(txt)
#         print(scores)  # For debugging in the terminal
        st.write(f"The predicted sentiment is {mapping[max_index]}, with a probability of {float(max_prob):.4f}")

        # Plot the results
        fig = px.pie(values=scores, names=labels, title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if option == "Fine-tuned roBERTa (for financial sentiments) warning: Model size is 1.4 GB, use only if you have the required space":
        scores, labels, max_prob, max_index, mapping = fine_tuned_roBERTa(txt)
#         print(scores)
        st.write(f"The predicted sentiment is {mapping[max_index.item()]}, with a probability of {float(max_prob.item()):.4f}")
        # Plot the results
        fig = px.pie(values=scores, names=labels, title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if option == "Compare Analyzers":
        # Analyze sentiment
        roBERTa_scores, roBERTa_labels,_ ,_ , _  = roBERTa_Analysis(txt)
#         print(roBERTa_scores)  # For debugging in the terminal

        NLTK_scores, NLTK_labels,_ ,_ , _  = NLTK_Analysis(txt)
#         print(NLTK_scores)  # For debugging in the terminal

        fine_tuned_roBERTa_scores, fine_tuned_roBERTa_labels,_ ,_ , _ = fine_tuned_roBERTa(txt)
#         print(fine_tuned_roBERTa_scores)  # For debugging in the terminal



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



