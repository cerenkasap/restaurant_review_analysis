## Restaurant Review - Sentiment Analysis 🌮: Project Overview

Pulled over **1000 examples** from Kaggle using pandas and opendatasets libraries in python.


### Code Used

Python version: *Python 3.7.11* 

Packages: *pandas, opendatasets, seaborn, matplotlib, numpy, nltk, wordcloud, collections, imblearn.over_sampling, re, string and textblob*

### Resources Used

[The dataset from Kaggle](https://www.kaggle.com/d4rklucif3r/restaurant-reviews)

[Functions for Text Data Cleaning](https://towardsdatascience.com/natural-language-processing-nlp-for-machine-learning-d44498845d5b)

## Data Collection
Used Kaggle to pull the datasets 5842 books with 2 columns:
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

