import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import nltk
from functions import load_data, prepare_and_clean_data, process_data
import argparse
import sys
import os
from joblib import load
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

def create_arg_parser():
    # Creates and returns the ArgumentParser object

    parser = argparse.ArgumentParser(description='Description of your app.')
    parser.add_argument('--textpath',
                    help='Path to test dataset (text).')
    parser.add_argument('--tagpath',
                    help='Path to test dataset (tag).')
    return parser


if __name__ == "__main__":
    np.random.seed(500)
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    path_text = parsed_args.textpath
    path_tags = parsed_args.tagpath
       
    text = load_data(path_text)
    tag = load_data(path_tags)
    data = prepare_and_clean_data(text,tag)

    data = process_data(data, "./stopwords/polish.stopwords.txt" )
    test_X = data['text_final']
    test_y = data['label']
    Tfidf_vect = TfidfVectorizer(max_features=3500)
    Tfidf_vect.fit(data['text_final'])
    Test_X_Tfidf = Tfidf_vect.transform(test_X)

    model = load("trained_model.pkl")

    RFC_pred = model.predict(Test_X_Tfidf)
    report = classification_report(test_y, RFC_pred)
    print(RFC_pred)
    print(report)
