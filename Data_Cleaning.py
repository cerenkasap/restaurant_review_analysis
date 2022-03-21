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
plt.savefig('images/length_of_reviews.png')
plt.show()

#Preprocessing

#Function to remove Punctuation
def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])# It will discard all punctuations
    return text_nopunct

df['Review']  = df['Review'].apply(lambda x: remove_punct(x))

# Function to Tokenize words
def tokenize(text):
    tokens = re.split('\W+', text) #W+ means that either a word character (A-Za-z0-9_) or a dash (-) can go there.
    return tokens

#We convert to lower as Python is case-sensitive. 
df['Review'] = df['Review'].apply(lambda x: tokenize(x.lower())) 

# All English Stopwords
stopword = nltk.corpus.stopwords.words('english')

# Function to remove Stopwords
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]# To remove all stopwords
    return text

df['Review'] = df['Review'].apply(lambda x: remove_stopwords(x))

#Stemming
ps = nltk.PorterStemmer()

def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

df['Review'] = df['Review'].apply(lambda x: stemming(x))

#Lemmatizer
wn = nltk.WordNetLemmatizer()

def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

df['Review'] = df['Review'].apply(lambda x: lemmatizing(x))



'''
#Text Cleaning
def Text_Cleaning(Text):    
    # Lowercase the texts
    Text = Text.lower()

    # Cleaning punctuations in the text
    punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    Text = Text.translate(punc)

    # Removing numbers in the text
    Text = re.sub(r'\d+', '', Text)

    # Remove possible links
    Text = re.sub('https?://\S+|www\.\S+', '', Text)

    # Deleting newlines
    Text = re.sub('\n', '', Text)

    return Text

#Text Preprocessing
# Stopwords
Stopwords = set(nltk.corpus.stopwords.words("english")) - set(["not"])

def Text_Processing(Text):    
    Processed_Text = list()
    Lemmatizer = WordNetLemmatizer()

    # Tokens of Words
    Tokens = nltk.word_tokenize(Text)

    # Removing Stopwords and Lemmatizing Words
    # To reduce noises in our dataset, also to keep it simple and still 
    # powerful, we will only omit the word `not` from the list of stopwords

    for word in Tokens:
            Processed_Text.append(Lemmatizer.lemmatize(word))
    return(" ".join(Processed_Text))

#Apply the functions
df['Review'] = df['Review'].apply(Text_Cleaning).apply(Text_Processing)


df. dropna()'''
df.to_csv('rest_review_data_cleaned.csv', index=False)
a=df.isnull().sum()