import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages-categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    '''The first visualization graph is the average message length for each feature_extraction
    The second visualization graph is the count of messages in each feature.
    '''
    df['message_length'] = df.message.apply(lambda x:len(x))
    featres_count = df.iloc[:,4:-1].sum().sort_values(ascending=False).values
    features_names = df.iloc[:,4:-1].sum().sort_values(ascending=False).index

    #Calculating the length of messages for each Feature
    classes_list = df.columns[4:-1]

    message_length_series=pd.Series()
    for col in classes_list:
        try:
            message_length_value = df.groupby(col).mean()['message_length'][1]
            message_length_series[col]= message_length_value
        except KeyError:
            #exception for any feature with 0 messages.
            message_length_series[col]= 0
    message_length_index = message_length_series.sort_values(ascending=False).index
    message_length_values = message_length_series.sort_values(ascending=False).values

    graphs = [
        {
            'data': [
                Bar(
                    x=message_length_index,
                    y=message_length_values
                )
            ],

            'layout': {
                'title': 'Average Messages Lengths of Each Feature',
                'yaxis': {
                    'title': "Message Length"
                },
                'xaxis': {
                    'title': "Feature Name"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=features_names,
                    y=featres_count
                )
            ],

            'layout': {
                'title': 'Count Of Messages In Each Feature',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Feature Name"
                }
            }
                }]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    #print(ids)
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    #print(graphJSON)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict_proba([query])
    classes_list = df.columns[4:-1]
    classification_results={}
    for i,prob in enumerate(classification_labels):
        if np.argmax(prob) >= 1:
            classification_results[f"{classes_list[i]}"]=np.max(prob)
        else:
            classification_results[f"{classes_list[i]}"] = 0

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
