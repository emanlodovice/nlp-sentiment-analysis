from nltk import word_tokenize, WordNetLemmatizer, classify
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold

import numpy
import data_parser
import stopwords
import re
import random


def feature_extractor(d):
    features = {}
    words = {}
    lematizer = WordNetLemmatizer()

    # get individual words from text
    words = [lematizer.lemmatize(word.lower()) for word in d.split(' ')]

    for word in words:
        word = word.encode('utf-8', 'ignore')
        if word != '':
            # check if word in not a stop word
            if word not in stopwords.stop_words:
                # check if the word is not a url or @person
                if not re.match('http://.*|@.*', word):
                    if word in features:
                        features[word] += 1
                    else:
                        features[word] = 1
    return features


def parse_data():
    data = data_parser.build_data()
    feature_set = [(feature_extractor(d['text']), d['class']) for d in data]
    random.shuffle(feature_set)
    training_set_size = int(len(feature_set) * 0.8)
    training_set = feature_set[:training_set_size]
    test_set = feature_set[training_set_size:]
    return training_set, test_set


def cross_validation(
        data_set, tfidf=TfidfTransformer(), nb=MultinomialNB(), n_folds=8):
    kf = KFold(len(data_set), n_folds=n_folds)
    best_classifier = None
    best_accuracy = 0
    for train, cv in kf:
        classifier = SklearnClassifier(
            Pipeline([('tfidf', tfidf), ('nb', nb)]))
        training_data = data_set[0:cv[0]] + data_set[cv[-1]:]
        cv_data = data_set[cv[0]:cv[-1]+1]
        classifier.train(training_data)
        accuracy = classify.accuracy(classifier, cv_data)
        best_classifier = classifier
        # if accuracy > best_accuracy:
        #     best_classifier = classifier
        #     best_accuracy = accuracy
    return best_classifier

training_set, test_set = parse_data()
print 'Training set size: ' + str(len(training_set))
print 'Test set size: ' + str(len(test_set))

# classifiers
# svm_classifier = SklearnClassifier(
#     Pipeline([('tfidf', TfidfTransformer()),
#               ('nb', LinearSVC())])).train(training_set)
# nb_classifier = SklearnClassifier(
#     Pipeline([('tfidf', TfidfTransformer()),
#               ('nb', MultinomialNB())])).train(training_set)
# lr_classifier = SklearnClassifier(
#     Pipeline([('tfidf', TfidfTransformer()),
#               ('nb', LogisticRegression())])).train(training_set)

# Get accuracy
lr_classifier = cross_validation(training_set)
lr_accuracy = classify.accuracy(lr_classifier, test_set)
print "Logistic Regression accuracy on test: " + str(lr_accuracy)
lr_accuracy_training = classify.accuracy(lr_classifier, training_set)
print "Logistic Regression accuracy on training: " + str(lr_accuracy_training)
