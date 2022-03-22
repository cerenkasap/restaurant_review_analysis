#import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#pull the cleaned data
df = pd.read_csv('rest_review_data_cleaned.csv')

#pull vectorized data
bag_df=pd.read_csv('bag_dfcsv')
tfidf_df=pd.read_csv('tfidf_df.csv')
ngram_df=pd.read_csv('ngram_df.csv')

#Model Selection
#Split the dataset
X=df[['Review_cleaned']]
y=df['Liked']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_bag = bag_df
X_tfidf = tfidf_df
X_ngram = ngram_df

#Split the dataset for vectorized datasets
X_bag_train, X_bag_test, y_bag_train, y_bag_test = train_test_split(X_bag, y, test_size=0.2, random_state=42)
X_tfidf_train, X_tfidf_test, y_tfidf_train, y_tfidf_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
X_ngram_train, X_ngram_test, y_ngram_train, y_ngram_test = train_test_split(X_ngram, y, test_size=0.2, random_state=42)


#Model building
dt = DecisionTreeClassifier()
lr = LogisticRegression()
svc = SVC()
rf = RandomForestClassifier()
Bayes = BernoulliNB()
KNN = KNeighborsClassifier()

Models = [dt, lr, svc, rf, Bayes, KNN]
Models_Dict = {0: "Decision Tree", 1: "Logistic Regression", 2: "SVC", 3: "Random Forest", 4: "Naive Bayes", 5: "K-Neighbors"}

for i, model in enumerate(Models):
  print("{} Test Accuracy in Bag of Words: {}".format(Models_Dict[i], cross_val_score(model, X_bag, y, cv = 10, scoring = "accuracy").mean()))
  print("{} Test Accuracy in TF-IDF: {}".format(Models_Dict[i], cross_val_score(model, X_tfidf, y, cv = 10, scoring = "accuracy").mean()))
  print("{} Test Accuracy in N-gram: {}".format(Models_Dict[i], cross_val_score(model, X_ngram, y, cv = 10, scoring = "accuracy").mean()))

#Hyperparamater Tuning
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
 
grid = GridSearchCV(svc, param_grid, refit = True, verbose = 3)
 
# fitting the model for grid search
grid.fit(X_tfidf_train, y_tfidf_train)
best_accuracy = grid.best_score_
best_parameters = grid.best_params_

print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

#Best Model
Classifier = SVC(C = 10, gamma=1, kernel = 'rbf')
Classifier.fit(X_tfidf_train, y_tfidf_train)
Prediction = Classifier.predict(X_tfidf_test)

#Metrics
accuracy_score(y_tfidf_test, Prediction)

#Confusion Matrix
ConfusionMatrix = confusion_matrix(y_tfidf_test, Prediction)

# Plotting Function for Confusion Matrix
colors = ['#4F6272', '#DD7596']

def plot_cm(cm, classes, title, normalized = False, cmap = plt.cm.BuPu):
    plt.imshow(cm, interpolation = "nearest", cmap = cmap)
    plt.title(title, pad = 20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    

    if normalized:
        cm = cm.astype('float') / cm.sum(axis = 1)[: np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Unnormalized Confusion Matrix")
  
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment = "center", color = "white" if cm[i, j] > threshold else "black")

    plt.tight_layout()
    plt.xlabel("Predicted Label", labelpad = 20)
    plt.ylabel("Real Label", labelpad = 20)
    
plot_cm(ConfusionMatrix, classes = ["Positive", "Negative"], title = "Confusion Matrix of Sentiment Analysis")
plt.tight_layout()
plt.savefig('images/confusion_matrix.png', dpi=300)