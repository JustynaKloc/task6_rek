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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import nltk
from imblearn.over_sampling import SMOTE
from joblib import dump
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
np.random.seed(500) #used to reproduce the same result every time
#!git clone https://github.com/bieli/stopwords.git  #repository with polish stopwords

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

text = load_data("./training_set_clean_only_text.txt")
labels = load_data("./training_set_clean_only_tags.txt")
data = prepare_and_clean_data(text,labels)
text_test = load_data("./test_set_only_text.txt")
labels_test = load_data("./test_set_only_tags.txt")
data_test = prepare_and_clean_data(text_test,labels_test)

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

data = process_data(data,"./stopwords/polish.stopwords.txt")
data_test = process_data(data_test,"./stopwords/polish.stopwords.txt")

train_X = data['text_final'] 
train_y = data['label']
test_X = data_test['text_final']
test_y = data_test['label']

Tfidf_vect = TfidfVectorizer(max_features = 3500)
Tfidf_vect.fit(data['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(train_X)
Test_X_Tfidf = Tfidf_vect.transform(test_X)

oversample = SMOTE(random_state=42)

Train_X_Tfidf,train_y = oversample.fit_resample(Train_X_Tfidf,train_y)


model = RandomForestClassifier()
model.fit(Train_X_Tfidf,train_y)

RFC_pred = model.predict(Test_X_Tfidf)
report = classification_report(test_y, RFC_pred, output_dict=True)
print(report)

dump(model,'task6model.joblib')
dump(report, 'report.pkl')
dump(RFC_pred, 'results.pkl')
