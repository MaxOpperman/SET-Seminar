import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, roc_auc_score, auc


def classify_sentiment(comments: [str], sentiment_rating, pickle_folder, train_set_size) -> [str]:
    # Remove punctuation and convert to lowercase
    # comments_processed = [(re.sub('[,/.!?:]', '', str(x))).lower() for x in comments]
    comments = [' ' if x is np.nan else x for x in comments]
    sentiment_label = [-1 if x < 0.2 else 1 if x > 0.2 else 0 for x in sentiment_rating]
    if not os.path.exists(f'./pickle/{pickle_folder}'):
        os.makedirs(f'./pickle/{pickle_folder}')
    x_train, x_test, y_train, y_test = train_test_split(comments, sentiment_label, random_state=0)
    if train_set_size is not None:
        x_train, x_test, y_train, y_test = train_test_split(comments, sentiment_label, random_state=0,
                                                            train_size=min(len(comments) - 1, train_set_size))
    # Fit the CountVectorizer to the training data
    vect = CountVectorizer().fit(x_train)
    # transform the documents in the training data to a document-term matrix
    x_train_vectorized = vect.transform(x_train)
    # Train the model
    model = LogisticRegression()

    lab_enc = preprocessing.LabelEncoder()
    encoded = lab_enc.fit_transform(y_train)
    print(f"Fitting model...")
    model.fit(x_train_vectorized, encoded)
    # Predict the transformed test documents
    print("Predicting...")
    predictions = model.predict(vect.transform(x_test))
    print('AUC: ', roc_auc_score(y_test, predictions))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    # Fit the CountVectorizer to the training data specifiying a minimum
    # document frequency of 5 and extracting 1-grams and 2-grams
    vect = CountVectorizer(min_df=5, ngram_range=(1, 2)).fit(x_train)
    x_train_vectorized = vect.transform(x_train)
    model = LogisticRegression()
    model.fit(x_train_vectorized, y_train)
    predictions = model.predict(vect.transform(x_test))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print('AUC: ', roc_auc_score(y_test, predictions))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
