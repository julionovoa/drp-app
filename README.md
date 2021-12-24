# Disaster response web app
This project shows a complete workflow to ingest data, clean, store in a database, for later train a NLP model to classify text messages into 36 available categories.

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

## Instructions
1. To re-create the results of this project, first clone this repository, and install the following Python packages:
* flask
* joblib
* nltk
* pandas
* plotly
* scikit-learn
* sqlalchemy

2. Using the terminal go to the folder **data** and execute the following command
```
python process_data.py disaster_messages.csv disaster_categories.csv DisastersDatabase.db
```
This step will read the two CSV files that contain the messages and their categories. The script will also clean and merge the datasets, and finally, it will create a SQLite database to store the cleaned data in the table `disasters`.

3. Using the terminal go to the folder **models** adn execute the following command:
```
python train_classifier.py DisastersDatabase.db disasters-response-model.pkl
```
This step will implement a text processing and machine learning pipeline. It will tune the parameters of the pipeline processes using `GridSearchCV`, and will create a serialized model called `disasters-response-model.pkl` that will be used in the web app to classify new text messages.

4. Using the terminal go to the folder **app** and execute the following command to start the Flask web app:
```
python run.py
```
Finally, follow the instructions shown in the terminal window to open a web browser and interact with the web app.
