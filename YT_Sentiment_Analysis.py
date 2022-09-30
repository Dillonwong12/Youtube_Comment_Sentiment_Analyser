import time
import re
import demoji
import spacy
import nltk
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords
from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from textblob.blob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd
pd.set_option('display.width', 2000)
pd.set_option('display.expand_frame_repr', False)


def scrape_comments(path):
    # Function to scrape youtube comments of video at URL of `path`

    data = []
    s = Service('/Users/dillonwong/Downloads/chromedriver')
    options = webdriver.ChromeOptions()
    options.add_argument('-headless')
    options.add_argument('-no-sandbox')
    options.add_argument('-disable-dev-shm-usage')

    with Chrome(service=s) as driver:
        try:
            wait = WebDriverWait(driver, 15)
            driver.get(path)

            for item in range(5):
                wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
                time.sleep(5)

            for comment in wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME,"style-scope ytd-comment-renderer"))):

                data.append(comment.text)

        except BaseException as e:
            print(e)
    return data


def clean_data(str1):
    # Cleans the scraped comments by removing emojis, numbers, symbols and white spaces

    demoji.download_codes()

    dem = demoji.findall(str1)
    for item in dem.keys():
        str1 = str1.replace(item, '')

    str1 = re.sub("[0-9]+", "", str1)
    str1 = ''.join(char for char in str1 if (char.isalpha() or char == ' '))

    return str1.lower().strip()


def remove_stopwords(all_comments):
    # Removes all commonly-used English words which have little to no effect on sentiment

    filtered_comments = []
    for comment in all_comments:

        filtered = " ".join([word.strip() for word in comment.split() if (word not in STOP_WORDS)])
        filtered_comments.append(filtered)

    return filtered_comments


def analyse_tb(comment):
    # Analyses sentiment based on the TextBlob package

    return TextBlob(comment).sentiment.polarity

def analyse_nltk(comment):
    # Analyses sentiment based on the NLTK package
    sia = nltk.SentimentIntensityAnalyzer()
    score = sia.polarity_scores(comment)["compound"]
    return score


def analyse_flair(comment):
    # Analyses sentiment based on the Flair package
    sentence = Sentence(comment)
    sia.predict(sentence)
    score = str(sentence.labels)

    num_score = "".join([char for char in score if char.isnumeric() or char == '.'])
    if 'NEG' in score:
        num_score = '-' + num_score
    return float(num_score)


def sent_anal(polarity):
    # Classifies polarity scores into "Positive", "Negative", and "Neutral"
    return 'Positive' if polarity > 0.25 else 'Negative' if polarity < -0.25 else 'Neutral'


def print_results(df):
    # Prints a short summary of the Sentiment Analysis results
    print("\nComments processed: " + str(len(df)))
    print("\nProportions:\n" + str(df['TB_Sentiment'].value_counts(normalize=True)))
    print(str(df['FL_Sentiment'].value_counts(normalize=True)))
    print("\nAverage TB Polarity: " + str(df['TB_Polarity'].mean()) + " - " + sent_anal(df['TB_Polarity'].mean()))
    print("Average FL Polarity: " + str(df['FL_Polarity'].mean()) + " - " + sent_anal(df['FL_Polarity'].mean()))


def create_df(data):
    # Takes `data`, the scraped YouTube comments, and produces a Pandas dataframe. First, filters entire comments to
    # retain only usernames and main text, then applies Sentiment Analysis to the `clean_data`.
    cleaned_comments = []
    comments =[]
    author = []
    for item in data:
        s = re.sub('\\n[0-9]+ (days|weeks|months|years|week|month|year|hours|hour|day|minutes|minute) ago', '', item)
        s = re.sub('\\nREPLY', '', s)
        s = re.sub(r'\.\\n[0-9][.]*[0-9]*[A-Z]*', '', s)
        t = s.split('\n')
        t[0] = re.sub(r'\(edited\)', '', t[0])
        author.append(t[0])
        s = "\n".join(t[1:])
        comments.append(t[1])
        s = clean_data(s)

        if len(s):
            cleaned_comments.append(s)

    cleaned_comments = remove_stopwords(cleaned_comments)
    df = pd.DataFrame(comments, index=author, columns=["Comments"])
    print(len(comments))
    print(len(cleaned_comments))
    df["Cleaned Comments"] = cleaned_comments
    df["TB_Polarity"] = df["Cleaned Comments"].apply(analyse_tb)
    df["TB_Sentiment"] = df["TB_Polarity"].apply(sent_anal)

    df["NTLK_Polarity"] = df["Cleaned Comments"].apply(analyse_nltk)
    df["NLTK_Sentiment"] = df["VD_Polarity"].apply(sent_anal)

    df["FL_Polarity"] = df["Cleaned Comments"].apply(analyse_flair)
    df["FL_Sentiment"] = df["FL_Polarity"].apply(sent_anal)

    # comment = df[df["Comments"].map(lambda text: TextBlob(text).detect_language()) != 'en']
    # authors = [author for author in comment.index]
    # df.drop(authors, inplace=True)

    print(df.head(20))
    return df


path = input("Enter video URL: ")
while not path.startswith("https://www.youtube.com/"):
    print("Please enter a valid YouTube video URL.")
    path = input("Enter video URL: ")

comments = scrape_comments(path)
sia = TextClassifier.load('en-sentiment')
comments_df = create_df(comments)
print_results(comments_df)