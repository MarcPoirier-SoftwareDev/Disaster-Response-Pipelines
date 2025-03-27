import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load data from the SQLite database.
    
    Args:
        database_filepath (str): Path to the SQLite database file.
    
    Returns:
        X (pd.Series): The messages (features).
        Y (pd.DataFrame): The categories (targets).
        category_names (list): List of category column names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Messages', engine) 
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize and clean the input text.
    
    Args:
        text (str): The text to be tokenized.
    
    Returns:
        list: A list of cleaned tokens.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def build_model():
    """
    Build a machine learning pipeline with GridSearchCV for hyperparameter tuning.
    
    Returns:
        GridSearchCV: The model pipeline with grid search.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators': [50, 100]
    }
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model and print classification reports for each category.
    
    Args:
        model: The trained model (GridSearchCV object).
        X_test (pd.Series): Test set messages.
        Y_test (pd.DataFrame): Test set categories.
        category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)
    for i, column in enumerate(category_names):
        print(f"Category: {column}")
        print(classification_report(Y_test[column], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the trained model to a pickle file.
    
    Args:
        model: The trained model (GridSearchCV object).
        model_filepath (str): Path to save the pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
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