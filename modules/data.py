import re
import nltk
import itertools
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from unicodedata import normalize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from spellchecker import SpellChecker

import sys
sys.path.append('/content/gdrive/MyDrive/Colab Notebooks/aed_projeto_4/modules/')
sys.path.append('/content/gdrive/MyDrive/Colab Notebooks/aed_projeto_4/')
sys.path.append('/content/gdrive/MyDrive/Colab Notebooks/')
from modules.acronymsInternet import acronymsInternet


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
PUNCTUATION_RE = re.compile(r'[^0-9a-z #+_]')
SMALL_WORDS_RE = re.compile(r'\W*\b\w{1,2}\b')
BLANKSPACE_RE = re.compile(r'\s{2,}')

STOPWORDS = stopwords.words('portuguese')
STOPWORDS.remove('muito')
STOPWORDS.remove('mais')
STOPWORDS.remove('mas')


def cleaning(text):
    cleanedText = wordBreaker(text)
    cleanedText = cleanedText.lower()
    cleanedText = reduceLengthening(cleanedText)
    cleanedText = cleanText(cleanedText)
    cleanedText = expandAcronyms(cleanedText)
    cleanedText = spellChecker(cleanedText)
    cleanedText = removeEmojify(cleanedText)
    cleanedText = removeSmallWords(cleanedText)
    return cleanedText


def wordBreaker(text):
    """
      Função para separar palavras juntadas. Ex.: UmExemplo.
    """
    textBreaked = ' '.join(re.findall('[A-Z][^A-Z]*', text.title()))    
    if textBreaked is not '':
        return textBreaked
    return text


def reduceLengthening(text):
    """
      Função para reduzir palavras alongadas. Ex.: Um Exemploooooo.
    """
    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))    
    return text


def spellChecker(text):
    """
      Função para correção ortográfica.
    """
    spell = SpellChecker(language='pt')
    return ' '.join([spell.correction(word) for word in text.split()])


def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = PUNCTUATION_RE.sub(' ', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text


def expandAcronyms(text):
    """
      Função para expansão de abreviaturas da internet.
    """
    return ' '.join([acronymsInternet.get(i, i) for i in text.split()])


def removeEmojify(text):
    regrex_pattern = re.compile(pattern = '['
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
                           ']+', flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


def removeSmallWords(text):
    text = SMALL_WORDS_RE.sub(' ', text)
    text = BLANKSPACE_RE.sub(' ', str(text))
    text = text.strip()
    return text


def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if not word in STOPWORDS]
    tokens = [word for word in tokens if word.isalpha()]
    return tokens


def printExample(idx, df):
    example = df[df.index==idx][['text', 'rating', 'stars']].values[0]
    
    if len(example) > 0:
        print(f'Text: {example[0]}')
        print(f'Rating: {example[1]}')
        print(f'Stars: {example[2]}')
        

def averageTokens(df, col):
    return df[col].apply(lambda x: len(x.split(' '))).mean()


def getTopNwords(corpus, ngrams=1, n=20): 
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
