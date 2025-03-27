import json
import plotly
import pandas as pd
import numpy as np  # Added for numerical operations

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
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
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Extract data needed for visuals
    # 1. Existing: Distribution of message genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # 2. New: Distribution of message categories
    category_names = df.columns[4:]  # Category columns start from index 4
    category_counts = df[category_names].sum().values  # Sum binary values to get counts
    
    # 3. New: Top 10 co-occurring category pairs
    co_occurrence = df[category_names].T.dot(df[category_names])  # Compute co-occurrence matrix
    upper_tri = co_occurrence.where(np.triu(np.ones(co_occurrence.shape), k=1).astype(bool))  # Upper triangle without diagonal
    co_occurrence_series = upper_tri.stack().sort_values(ascending=False)  # Flatten and sort
    top_10 = co_occurrence_series.head(10)  # Top 10 pairs
    pair_names = [f"{cat1} & {cat2}" for cat1, cat2 in top_10.index]  # Format pair names
    pair_counts = top_10.values  # Corresponding counts
    
    # Create visuals
    graphs = [
        # Existing visualization: Genre distribution
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': 'Genre'}
            }
        },
        # New visualization 1: Category distribution
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': 'Category', 'tickangle': -45},  # Rotate labels for readability
                'width': 1200,  # Wider plot to fit many categories
                'height': 600
            }
        },
        # New visualization 2: Top co-occurring category pairs
        {
            'data': [
                Bar(
                    x=pair_names,
                    y=pair_counts
                )
            ],
            'layout': {
                'title': 'Top 10 Co-occurring Category Pairs',
                'yaxis': {'title': 'Number of Messages'},
                'xaxis': {'title': 'Category Pair', 'tickangle': -45}  # Rotate labels for readability
            }
        }
    ]
    
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the go.html template
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()