import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump
from functions import load_data, prepare_and_clean_data, process_data
np.random.seed(500) #used to reproduce the same result every time
#!git clone https://github.com/bieli/stopwords.git  #repository with polish stopwords

if __name__ == "__main__":

    text = load_data("./training_set_clean_only_text.txt")
    labels = load_data("./training_set_clean_only_tags.txt")
    data = prepare_and_clean_data(text,labels)
    data = process_data(data,"./stopwords/polish.stopwords.txt")
    train_X = data['text_final'] 
    train_y = data['label']

    Tfidf_vect = TfidfVectorizer(max_features=3500)
    Tfidf_vect.fit(data['text_final'])
    Train_X_Tfidf = Tfidf_vect.transform(train_X)

    oversample = SMOTE(random_state=42)

    Train_X_Tfidf,train_y = oversample.fit_resample(Train_X_Tfidf, train_y)

    model = RandomForestClassifier()
    model.fit(Train_X_Tfidf,train_y)

    dump(model,'trained_model.pkl')
