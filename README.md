### Restaurant Review - Sentiment Analysis 🌮: Project Overview

Pulled over **1000 examples** from Kaggle using pandas and opendatasets libraries in python.


### Code Used

Python version: *Python 3.7.11* 

### Resources Used

[The dataset from Kaggle](https://www.kaggle.com/d4rklucif3r/restaurant-reviews)


## Data Collection
Used Kaggle to pull the datasets 5842 books with 2 columns:
* Review             
* Liked            


## Data Cleaning

After pulling the data, I cleaned up the dataset to reduce noises in the dataset. The changes were made follows:

* Made lowercase the sentences, cleaned punctuations in the sentences, deleted the newlines, removed numbers and possible links from the sentences.
* Removed stop words from the sentences and lemmatized them.
