#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
nltk.download('stopwords')

#pull the dataset 
df=pd.read_csv('./restaurant-reviews/Restaurant_Reviews.tsv', sep='\t', index_col=False)

#Check if there is any missing data
df.isnull().sum()
df.info()

#Create a bar graph for Liked column
liked=df["Liked"].value_counts().plot(kind = "bar", color = "salmon")
plt.title("Amount of Reviews", pad = 20)
plt.xlabel("Liked", labelpad = 15)
plt.ylabel("Amount of Reviews",labelpad = 20)
plt.tight_layout()
plt.savefig('images/l_of_reviews.png')
plt.show()

#Preprocessing

#Lowercase the texts
def lowercase_text(text):
    text=text.lower()
    return text

df['Review_cleaned']  = df['Review'].apply(lambda x: lowercase_text(x))

#Function to remove Punctuation
def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])# It will discard all punctuations
    return text_nopunct

df['Review_cleaned']  = df['Review_cleaned'].apply(lambda x: remove_punct(x))

# Function to Tokenize words
def tokenize(text):
    tokens = re.split('\W+', text) #W+ means that either a word character (A-Za-z0-9_) or a dash (-) can go there.
    return tokens

#We convert to lower as Python is case-sensitive. 
df['Review_cleaned']  = df['Review_cleaned'] .apply(lambda x: tokenize(x.lower())) 

# Function to remove Stopwords
# All English Stopwords
stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]# To remove all stopwords
    return text

df['Review_cleaned'] = df['Review_cleaned'] .apply(lambda x: remove_stopwords(x))

#Lemmatizer
wn = nltk.WordNetLemmatizer()

def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

df['Review_cleaned']  = df['Review_cleaned'] .apply(lambda x: lemmatizing(x))

#Save the cleaned dataset
df.to_csv('rest_review_data_cleaned.csv', index=False)
