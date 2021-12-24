# Disaster response web app

## Repository structure
```
- app
| - template
| |- master.html
| |- go.html
|- run.py

- data
|- disaster_categories.csv
|- disaster_messages.csv
|- process_data.py
|- DisastersDatabase.db

- models
|- train_classifier.py
|- disasters-response-model.pkl 

- README.md
```

## Project Components
The project was organized into three components:

1. ETL Pipeline
In the **data** folder in this repository one can find the input datasets (\*.csv), the data load and data cleaning Python script (process_data.py), and the SQLite database (DisastersDatabase.db) that stores the cleaned data.

2. ML Pipeline
The folder **models** contains the Python script (train_classifier.py) that implements a text processing and machine learning pipeline to classify text messages into one or more categories. Once the model was trained, a serialed model (disasters-response-model.pkl) was created.

3. Flask Web App
In the folder **app** there are two web templates for the main app webpage and the app results webpage. The Python script (run.py) provides the code to execute the app using Flask as backend.