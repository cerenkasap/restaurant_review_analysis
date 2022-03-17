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
        if word not in Stopwords:            
            Processed_Text.append(Lemmatizer.lemmatize(word))            
    return(" ".join(Processed_Text))

#Apply the functions
df['Review'] = df['Review'].apply(Text_Cleaning).apply(Text_Processing)

df.to_csv('rest_review_data_cleaned.csv', index=False)