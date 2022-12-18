import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

def extract_text_features(data):
    features, scores = [], []
    for i in range(len(data)):
        cur_row = data.iloc[i]
        cur_text = []

        if not type(cur_row['tag']) == float:
            cur_text.append(cur_row['tag'])
        if not type(cur_row['title']) == float:
            ## may need to do some text cleaning here
            tokenize_title = " ".join(word_tokenize(cur_row['title']))
            cur_text.append(tokenize_title)
        if not type(cur_row['body']) == float:
            tokenize_body = " ".join(word_tokenize(cur_row['body']))
            cur_text.append(tokenize_body)
        features.append(" ".join(cur_text))
        scores.append(cur_row['score'])
    return features, scores

def create_splits(features, scores, test_size = 0.10):
    features_train, features_test, scores_train, scores_test = train_test_split(features, scores, test_size=test_size, random_state=7)
    return features_train, scores_train, features_test, scores_test


def regression_model(X_train_svd, y_train, X_test_svd, y_test):
    ## train a baseline model with default params

    print("\nModel training started ...")

    model = RandomForestRegressor(n_estimators = 100, n_jobs=-1, verbose=1)
    model.fit(X_train_svd, y_train)

    print("\nModel training finished ...")

    ## make prediction

    y_pred = model.predict(X_test_svd)
    y_pred_train =  model.predict(X_train_svd)

    return model, y_pred, y_pred_train

## read data

data = pd.read_csv('askscience_data.csv', sep = ',')
data = data.drop_duplicates(subset=['title'])
print("\nDone reading data ...")

## extract text features, tokenize

features, scores = extract_text_features(data)

print("\nDone extracting text features ...")

## create train, test splits

features_train, scores_train, features_test, scores_test = create_splits(features, scores)

print("\nDone creating train/test splits ...")

vectorizer = TfidfVectorizer(stop_words = 'english')
vectorizer = vectorizer.fit(features_train)

print("\nDone fitting a vectorizer ...")

svd = TruncatedSVD(n_components=100)

## features and labels preparation

X_train = vectorizer.transform(features_train)
X_test = vectorizer.transform(features_test)

## Use dimensionality reduction, noise removal

svd = svd.fit(X_train)

## use the new svd transform features

X_train_svd = svd.transform(X_train)
X_test_svd = svd.transform(X_test)

print("\nDone transform using svd ...")

y_train = scores_train
y_test = scores_test

model, y_pred, y_pred_train = regression_model(X_train_svd, y_train, X_test_svd, y_test)

## evaluate the predictions on the test set
error_metric_train = mean_squared_error(y_train, y_pred_train, squared = False)
error_metric_test = mean_squared_error(y_test, y_pred, squared = False)

print("RMSE on train set: ", error_metric_train)
print("RMSE on test set: ", error_metric_test)

## write predictions
f = open('predictions.txt', 'w')
for i in range(len(y_test)):
    f.write(features_test[i])
    f.write('\t')
    f.write(str(y_test[i]))
    f.write('\t')
    f.write(str(y_pred[i]))
    f.write('\n')
f.close()


'''

Some stats

Tag distribution of 100 posts with the most scores:

Biology           16
Physics           14
Earth Sciences    13
Astronomy          9
Human Body         9
COVID-19           8
Medicine           7
Engineering        7
Chemistry          3
Computing          3
Planetary Sci.     2
Linguistics        2
Social Science     2
Mathematics        1
Psychology         1
Anthropology       1
Meta               1

Average word lengths of title + body for the top 100 posts is 72 words 

Average score for the top 100 posts is 21167.1

Some frequent words (excluding stop words) in top 100 posts:

('people', 20)
('black', 19)
('years', 14)
('water', 13)
('vaccine', 9)
('coronavirus', 9)
('gravitational', 9)
('waves', 9)
('STEM', 9)
('COVID-19', 7)
('cancer', 7)
('space', 7)
('light', 7)
('medical', 7)

'''






