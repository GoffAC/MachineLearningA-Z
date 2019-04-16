# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset (it is a TSV)
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts (doing this manually gives better understanding and tighter control as well as  learning)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords #remove irrelevant words for understanding sentiment (like "this")
#to see the list of stop words run the code below
"""stopWords = set(stopwords.words('english'))
print(stopWords)
"""

from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #substitute punctuation with a space
    review = review.lower() #lowercase all so that capitalisation doesn't infer meaning
    review = review.split() #turns string into an array of words
    ps = PorterStemmer() #create an object to find root words - love === loved
    #for syntax explanation in this for statement: 
        #https://stackoverflow.com/questions/6475314/python-for-in-loop-preceded-by-a-variable
    #essentially  review = [function(number) for number in numbers if condition(number)]
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #join back into text, with a space seperator    
    review = ' '.join(review) 
    corpus.append(review)

# Creating the Bag of Words model
#This makes a large table of columns relating to new words and rows of each entry
#As it will be a very empty set it will be high in sparcity
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # filters so that we have the most common 1500 columns/words
X = cv.fit_transform(corpus).toarray()
#bring in training data (whether it is a postitive or negative review)
y = dataset.iloc[:, 1].values

#Copied from Niave Bayes without feature scaling
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)