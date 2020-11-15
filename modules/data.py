import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unicodedata import normalize
from sklearn.feature_extraction.text import CountVectorizer

from modules.acronymsInternet import acronymsInternet

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
PUNCTUATION_RE = re.compile(r'[^0-9a-z #+_]')
STOPWORDS = stopwords.words('portuguese')

def clean_text(text):
    text = BeautifulSoup(text, "lxml").text
    text = text.lower()
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = PUNCTUATION_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text


def expandAcronyms(text):
    return ' '.join([acronymsInternet.get(i, i) for i in text.split()])


def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if not word in STOP_WORDS]
    tokens = [word for word in tokens if word.isalpha()]
    return tokens


def print_example(idx, df):
    example = df[df.index==idx][['text', 'rating', 'stars']].values[0]
    
    if len(example) > 0:
        print(f'Text: {example[0]}')
        print(f'Rating: {example[1]}')
        print(f'Stars: {example[2]}')
        

def average_tokens(df, col):
    return df[col].apply(lambda x: len(x.split(' '))).mean()


def get_top_n_words(corpus, ngrams=1, n=20): 
    if ngrams == 3:
        vec = CountVectorizer(ngram_range=(3, 3), stop_words=STOPWORDS).fit(corpus)    
    elif ngrams == 2:
        vec = CountVectorizer(ngram_range=(2, 2), stop_words=STOPWORDS).fit(corpus)    
    else:
        vec = CountVectorizer(stop_words=STOPWORDS).fit(corpus)
    
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]