import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from unicodedata import normalize

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
PUNCTUATION_RE = re.compile(r'[^0-9a-z #+_]')
STOPWORDS = stopwords.words('portuguese')


def print_example(idx, df):
    example = df[df.index==idx][['text', 'rating', 'stars']].values[0]
    
    if len(example) > 0:
        print(f'Text: {example[0]}')
        print(f'Rating: {example[1]}')
        print(f'Stars: {example[2]}')
        
        
def average_tokens(df, col):
    return df[col].apply(lambda x: len(x.split(' '))).mean()


def clean_text(text):
    text = BeautifulSoup(text, "lxml").text
    text = text.lower()
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = PUNCTUATION_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text