import sys
import pandas as pd
import re
import joblib
from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("punkt")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report



def load_data(database_filepath):
    """Load the 'disasters' table from the disasters database
    and return dataframes: X (messages), Y (categories), and
    the list categories_names"""

    # Create SQLite engine
    engine = create_engine("sqlite:///" + database_filepath)
    
    # Load data and create dataframe
    df = pd.read_sql("SELECT * FROM disasters", engine)
    
    # Slipt dataframe into X, Y, category_names
    X = df.loc[:, "message"].values
    Y = df.drop(["id", "message", "original", "genre", "child_alone"], axis=1)
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """Tokenize and lemmatize the messages"""

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Remove stop words
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]

    # Reduce words to their root form
    tokens = [WordNetLemmatizer().lemmatize(w) for w in words]

    return tokens


def build_model():
    """Build a machine learning pipeline to classify messages into
    any of the available categories"""

    # Build a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression()))
    ])

    # GridSearch parameters
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__max_iter': [500, 1000]
    }

    # create GridSearch object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model and print the prediction accuracy"""

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """Serialize the model"""
    
    joblib.dump(model, model_filepath)


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
