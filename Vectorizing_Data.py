#import libraries
import pandas as pd
import re
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#pull the dataset
df=pd.read_csv('rest_review_data_cleaned.csv')

#Vectorizing Data
#Bag-Of-Words
count = CountVectorizer()
bag_df=count.fit_transform(df['Review_cleaned'])
bag_df = pd.DataFrame(bag_df.toarray(), columns=count.get_feature_names())
bag_df.to_csv('bag_dfcsv', index=False)

#TF-IDF
tfidf = TfidfVectorizer()
tfidf_df = tfidf.fit_transform(df['Review_cleaned'])
tfidf_df = pd.DataFrame(tfidf_df.toarray(), columns=tfidf.get_feature_names())
tfidf_df.to_csv('tfidf_df.csv', index=False)

#N-Grams
# It applies only bigram vectorizer
ngram = CountVectorizer(ngram_range=(2,2))
ngram_df = ngram.fit_transform(df['Review_cleaned'])
ngram_df = pd.DataFrame(ngram_df.toarray(), columns=ngram.get_feature_names())
ngram_df.to_csv('ngram_df.csv', index=False)
