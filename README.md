# Disaster Response Pipeline Project

### Motivation:
In a disaster like floods, storm, hurricanes...etc. The quick response for needs of the people can save a lot of lives.
With the help of machine learning, the messages on social media platforms can be analyzed and classified so that the messages
can be directed to the responsible organizations to offer the quick support to the victims.
This project is based on machine natural language processing (NLP) and machine learning algorithms to analyze the text dataset
and predict the category of each message.

### Installation:
The required packages for installation of the dependencies and packages are listed in `requirements.txt` file.
For installation of the packages run the following command in your terminal.
`pip install -r requirements.txt`

### Contents:
The structure of the files is as below.
```
  -app
  | - template
  | |- master.html  # main page of web app
  | |- go.html  # classification result page of web app
  | |- run.py  # Flask file that runs app

  -data
  |- disaster_categories.csv  # data to process
  |- disaster_messages.csv  # data to process
  |- process_data.py
  |- DisasterResponse.db   # database to save clean data to

  -models
  |- train_classifier.py
  |- classifier1.pkl  # saved model
  -screenshots
  |-index-home-page.jpg
  |-Result-page.jpg
  |-graphs.jpg
  -README.md
  -LICENSE.txt
  -requirements.txt
  -ML Pipeline Preparation.ipynb
  -ETL Pipeline Preparation.ipynb
```
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
![alt text](https://github.com/bassemessam/disaster-response-project/blob/main/screenshots/index-page.jpg?raw=true)

4. Classify message button will direct you to the results page. The classification is done for each category with the Certainty rate of the classification.

![alt text](https://github.com/bassemessam/disaster-response-project/blob/main/screenshots/result-page.jpg?raw=true)

5. In web browser, you will find two visualization graphs from the dataset.
  - Count of messages in each category in the dataset.
  - The average message length in each category.


### License:
The datasets used in this project are created by Figure Eight and provided a supervised dataset with different categories labels. Also, the project created as a part of Udacity data science nanodegree.
This project is created under MIT license. The details are included in `LICENSE.txt`
