import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle


def load_data(database_filepath):
    '''A function to load the data from databse file and split to X and Y.
    Args:
    - database_filepath: A string of the dataset file path.
    Returns:
    X: numpy array of dependent variable.
    Y: numpy array of independent variable (classification variables)
    category_names: A list of classification categories.
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name = 'messages-categories',con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''A function to split the text data to tokens.
    Args:
    - text: A string of text dataset.
    Returns:
    tokens: strings of cleaned tokens after removing stop words and lemmatization.
    '''
    text = re.sub(r"[^A-Za-z0-9]", " ",text)
    words = word_tokenize(text.lower().strip())
    tokens = [word for word in words if word not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    return tokens


def build_model(X_train, Y_train):
    '''A function to train the model and select the best parameters of the model.
    Args:
    - X_train: The training dataset.
    - Y_train: The training classification of the features.
    Returns:
    best_parameters: A model with best parameters.
    '''
    pipeline = Pipeline ([('vect',CountVectorizer(tokenizer=tokenize)),
                          ('tfidf',TfidfTransformer()),
                          ('clf',MultiOutputClassifier(RandomForestClassifier()))
                           ])
    '''Parameters to be used to get the best estimator, but it needs a lot of
    computational resources. The parameters reduced for the code testing puposes

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [int(x) for x in np.linspace(50,1000,num=20)],
        'clf__estimator__min_samples_split': [2, 3, 4,5,15,100],
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__max_depth': [int(x) for x in np.linspace(5,30,num=6)],
        'clf__estimator__min_samples_leaf': [1, 2, 5, 10],
        'clf__estimator__max_features': ['auto','sqrt']
    }'''
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_features': (5000, 10000),
        'tfidf__use_idf': (True, False)}

    cv = GridSearchCV(pipeline, param_grid=parameters,verbose=2)
    cv.fit(X_train,Y_train)
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, cv.best_estimator_.get_params()[param_name]))

    best_parameters = cv.best_estimator_
    return best_parameters



def evaluate_model(model, X_test, Y_test, category_names):
    '''A function to predict the data using the model and evaluate the performance.
    Args:
    - model: An object of the model.
    - X_test: test data of the dependent variable.
    - Y_test: test data of classification categories.
    Returns:
    Prints the classification reports of each class.
    '''
    y_pred = model.predict(X_test)
    for i,col in enumerate(category_names):
        print(f"The classification report for {col} is \n {classification_report(Y_test[col],y_pred[:,i])}")


def save_model(model, model_filepath):
    '''A function to save the model to .pkl file.
    Args:
    - model: An object of the model.
    - model_filepath: A string of the model .pkl file.
    Returns:
    -None
    '''
    file = open("{}".format(model_filepath),'wb')
    pickle.dump(model,file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model(X_train, Y_train)

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
