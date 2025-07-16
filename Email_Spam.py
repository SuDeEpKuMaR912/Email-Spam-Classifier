import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from joblib import load, dump

#LOAD THE DATASET
df= pd.read_csv("spam.csv")

'''print(df.head())
print(df.groupby('Category').describe())'''

#CONVERTING TEXT TO NUMBERS: CATEGORY
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)

#print(df.head())

#SPLITING TRAINING AND TESTING DATA
x_train, x_test, y_train, y_test= train_test_split(df.Message, df.spam, test_size=0.25)

#CONVERTING TEXT IN 'Message' COLUMN USING COUNT VECTORIZER TECHNIQUE
v= CountVectorizer()
x_train_count= v.fit_transform(x_train.values)

#print(x_train_count.toarray()[:3])

#TRAINING MODEL USING MULTINOMIAL NAIVE BAYES: SINCE WE HAVE DISCRETE DATA
model= MultinomialNB()
model.fit(x_train_count, y_train)

#PREDICT EMAILS
emails= [
    'Hey mohan, can we get together to watch football game tomorrow?',
    'Upto 20 per cent discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count= v.transform(emails)

predictions= model.predict(emails_count)
#print(predictions)           #[0 1]

#ACCURACY TEST
x_test_count= v.transform(x_test)
#print(model.score(x_test_count, y_test))         #0.9849246231155779


#####OPTIONAL#####
#ALTERNATIVELY WE CAN SIMPLIFY ALL THIS STEP WITH PIPELINE
clf= Pipeline([
    ('vectorize', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(x_train, y_train)         #we can directly train using x_train which is not initially vectorized

print(clf.predict(emails))        #[0 1]
