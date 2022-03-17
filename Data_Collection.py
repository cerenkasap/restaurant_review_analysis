#import libraries
import pandas as pd
import opendatasets as od

#Download the data and save as df
od.download("https://www.kaggle.com/d4rklucif3r/restaurant-reviews")

#Pull the data
df=pd.read_csv('./restaurant-reviews/Restaurant_Reviews.tsv', sep='\t', index_col=False)