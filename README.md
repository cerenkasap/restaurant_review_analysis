## Restaurant Review - Sentiment Analysis 🌮: Project Overview

Created a model that can classify a Restaurant Review as a Positive or a Negative review with **(77% Accuracy)** to detect polarity within the text.

Pulled over **1000 examples** from Kaggle using pandas and opendatasets libraries in python.

Applied **Logistic Regression, Support Vector Classifier, Random Forest Classifier, Bernoulli Naive Bayes**, and **KNeighborsClassifier** and optimized using **GridSearchCV** to find the best model.

### Code Used

Python version: *Python 3.7.11* 

Packages: *pandas, opendatasets, seaborn, matplotlib, numpy, nltk, wordcloud, collections, imblearn.over_sampling, re, string and textblob*

### Resources Used

[The dataset from Kaggle](https://www.kaggle.com/d4rklucif3r/restaurant-reviews)

[Functions for Text Data Cleaning](https://towardsdatascience.com/natural-language-processing-nlp-for-machine-learning-d44498845d5b)

 
## Data Collection
Used Kaggle to pull the datasets 1000 reviews with 2 columns:
* Review             
* Liked            

![alt text](https://github.com/cerenkasap/restaurant_review_analysis/blob/master/images/l_of_reviews.png "Length of Reviews on Raw Data")


## Data Cleaning

After pulling the data, I cleaned up the dataset to reduce noises in the dataset. The changes were made follows:

* Made lowercase the sentences, removed punctuations in the sentences, tokenized words, removed stop words from the sentences and lemmatized them.



## Exploratory Data Analysis

Visualized the cleaned data to see the trends.

* Created *WordCloud* for **Reviews**.
![alt text](https://github.com/cerenkasap/restaurant_review_analysis/blob/master/images/wordcloud.png "Word Cloud")

* Created *Donut chart* for **Review** data.
![alt text](https://github.com/cerenkasap/restaurant_review_analysis/blob/master/images/donut_chart.png "% of sentiments")
It looks like our data is balanced.

* Created *2-Gram Analysis Bar Graphs* for **Review** variables.
![alt text](https://github.com/cerenkasap/restaurant_review_analysis/blob/master/images/p_2gram.png "2-gram of Reviews with Positive Reviews")
![alt text](https://github.com/cerenkasap/restaurant_review_analysis/blob/master/images/n_2gram.png "2-gram of Reviews with Negative Reviews")


* Created a histogram for **Polarity Score** in Sentences
![alt text](https://github.com/cerenkasap/restaurant_review_analysis/blob/master/images/polarity_score.png "Polarity Score in Sentences")
Sentences with *negative* polarity will be in range of [-1, 0), *neutral* ones will be 0.0, and *positive* reviews will have the range of (0, 1).

* Created a histogram for **Length of Sentences** 
![alt text](https://github.com/cerenkasap/restaurant_review_analysis/blob/master/images/length_of_reviews.png "Length of Reviews")
Based on this histogram, we know that our review has text length between approximately 20-80 characters.

* Created a histogram for **Word Counts** in Sentences
![alt text](https://github.com/cerenkasap/restaurant_review_analysis/blob/master/images/word_counts.png "Word Counts in Reviews")
From the figure above, we infer that most of the reviews consist of 1 word to 10 words. 

## Feature Extraction (Vectorization)

Created text features with *Term Frequency - Inverse Document Frequency (TF-IDF), Bag-of-Words, and N-Gram* then saved them in different dataframes.


## Model Building

Data were split into **train (80%)** and **test (20%)** sets.

## Model Performance Evalution
I used six models *(Decision Tree Classifier, Logistic Regression, Support Vector Classifier, Random Forest Classifier, Bernoulli Bayes, and KNeighborsClassifier)* to predict the sentiment and evaluated them by using *Cross Validation Accuracy Score* with three different vectorized data. 

I applied cross_val_score to different model with vectorized data combinations to choose the model with the best accuracy score. 

Logistic Regression model with TF-IDF vectorized data performed better than any other models in this project.


|Model                                       |Cross Validation Accuracy Score|  
| -------------                              |:-----------------:|                       
|Decision Tree with Bag of Words data         |0.7   |
|Decision Tree with TF-IDF data               |0.7025|
|Decision Tree with N-gram data               |0.5800|
|Logistic Regression with Bag of Words data   |0.7762|
|Logistic Regression with TF-IDF data         |0.7938|
|Logistic Regression with N-gram data         |0.5713|
|SVC with Bag of Words data                   |0.7775|
|SVC with TF-IDF data                         |0.7863|
|SVC with N-gram data                         |0.58  |
|Random Forest with Bag of Words data         |0.7475|
|Random Forest with TF-IDF data               |0.7613|
|Random Forest with N-gram data               |0.5763|
|Naive Bayes with Bag of Words data           |0.7562|
|Naive Bayes with TF-IDF data                 |0.7562|
|Naive Bayes with N-gram data                 |0.5725|
|K-Neighbors with Bag of Words data           |0.6788|
|K-Neighbors with TF-IDF data                 |0.7263|
|K-Neighbors with N-gram data                 |0.5163|


## Hyperparameter Tuning

We got the best accuracy **79.12%** with GridSearchCV and find the optimal hyperparameters.

## Best Model

Applied Logistic Regression model with the optimal hyperparameters and got **77%** Test Accuracy score.

## Confusion Matrix
![alt text](https://github.com/cerenkasap/restaurant_review_analysis/blob/master/images/confusion_matrix.png "Confusion Matrix of Restaurent Review Analysis")

The Confusion Matrix above shows that our model needs to be improved to categorize reviews better.

Since the accuracy on the training data **(79%)** is higher than the accuracy on the test data **(77%)**, we can say our model is **overfitting** and needs to be improved.

Thanks for reading :) 

