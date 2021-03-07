import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import unicodedata
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode()

def cleaning_pipeline(corpus, cleaner=WordNetLemmatizer):
    
    corpus = [row.lower() for row in corpus]
    print('Lowercase:')
    print(f'{corpus[:1]}')
    
    corpus = [punctuation_removal(row) for row in corpus]
    print('\nPunctuation Removed:')
    print(f'{corpus[:1]}')
    
    stop = stopwords.words('english')
    corpus = [' '.join([word for word in row.split() if word not in (stop)]) for row in corpus]
    print('\nStopwords Removed:')
    print(f'{corpus[:1]}')
    
    corpus = [remove_accents(row) for row in corpus]
    print('\nAccents Removed:')
    print(f'{corpus[:1]}')
    
    word_list = [word_tokenize(row) for row in corpus]
    print('\nTokenized:')
    print(f'{word_list[:1]}')
    
    stem_lemm = cleaner()
    stem_lemm_output = [' '.join([stem_lemm.lemmatize(word) for word in words]) for words in word_list]
    print('\nLemmatized:')
    print(f'{stem_lemm_output[:1]}')
    
    return stem_lemm_output