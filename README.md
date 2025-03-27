# Disaster Response Pipeline Project

### Project Summary

This project, developed as part of the Udacity Data Scientist Nanodegree, focuses on building a machine learning pipeline to classify disaster response messages. The objective is to create a web application where users can input a disaster-related message and receive classification results across 36 predefined categories. This tool aims to assist disaster relief agencies by enabling them to quickly identify and prioritize critical needs based on message content. The project integrates an ETL (Extract, Transform, Load) pipeline for data preprocessing, a machine learning pipeline for training a multi-output classifier, and a Flask-based web app for user interaction.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage (if using a workspace like Udacity's). Alternatively, open a browser and navigate to http://0.0.0.0:3001/ to access the app.

### Files in the Repository

- data/: Directory containing raw data and the ETL pipeline script.
	- disaster_messages.csv: CSV file with raw disaster messages.

	- disaster_categories.csv: CSV file containing category labels corresponding to the messages.

	- process_data.py: Python script that performs the ETL process—loading raw data, cleaning it, and storing it in an SQLite database.

	- DisasterResponse.db: SQLite database file generated by process_data.py, storing the cleaned and preprocessed data.

- models/: Directory for the machine learning pipeline and trained model.
	- train_classifier.py: Python script that loads data from DisasterResponse.db, trains a multi-output classifier, and saves it as a pickle file.

	- classifier.pkl: Pickle file containing the trained classifier model, generated by train_classifier.py.

- app/: Directory containing the Flask web application files.
	- run.py: Main script to launch the web app, handling user inputs and displaying classification results.

	- templates/:
		- master.html: HTML template for the app’s homepage, including data visualizations and input fields.

		- go.html: HTML template for displaying the classification results of a user-submitted message.

- README.md: This file, providing an overview of the project, instructions, and file descriptions.


### Additional Notes

- To customize the project, you can modify the ETL pipeline in process_data.py, the ML model in train_classifier.py, or the web app in run.py and its templates.






