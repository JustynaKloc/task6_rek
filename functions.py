import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_data(path: str)-> list:
  dataset = []
  with open(path) as file:
    for line in file:
      dataset.append(line)
  return dataset 

def prepare_and_clean_data(text: list, labels: list):
  df1 = {'label': labels, 'text': text}
  data = pd.DataFrame(df1)
  data = data.replace('\n','', regex=True)
  data["label"] = pd.to_numeric(data["label"])
  #remove @annonymized_account 
  data = data.replace('@anonymized_account','', regex=True)
  return data

def process_data(data: pd.DataFrame, stopwords_path: str):
  # remove blank rows.
  data['text'].dropna(inplace=True)
  # Change text to lower case
  data['text'] = [entry.lower() for entry in data['text']]
  # Tokenization 
  data['text']= [word_tokenize(entry) for entry in data['text']]
  # Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
  tag_map = defaultdict(lambda : wn.NOUN)
  tag_map['J'] = wn.ADJ
  tag_map['V'] = wn.VERB
  tag_map['R'] = wn.ADV
  stopwords = open(stopwords_path,'r').read().split('\n')
  for index,entry in enumerate(data['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag', if the word is N, V, ADV, ADJ
    for word, tag in pos_tag(entry):
        #check for Stop words and consider only alphabets
        if word not in stopwords and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    data.loc[index,'text_final'] = str(Final_words)
  return data
