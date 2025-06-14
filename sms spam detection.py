#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[3]:


# Step 1: Load Dataset (CSV format)
df = pd.read_csv('spam.csv', encoding='latin1')


# In[4]:


# Keep only the required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']  # Rename columns for clarity


# In[5]:


# Step 2: Drop missing messages
df = df.dropna(subset=['message'])


# In[6]:


# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# In[7]:


# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)


# In[8]:


# Step 4: TF-IDF Vectorization
X_train = X_train.fillna("").astype(str)
X_test = X_test.fillna("").astype(str)
tfidf = TfidfVectorizer(stop_words='english', max_df=0.9)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# In[9]:


# Step 5: Train and Evaluate
def train_and_evaluate(model, name):
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


# In[19]:


# Models
train_and_evaluate(MultinomialNB(), "Naive Bayes")
train_and_evaluate(LogisticRegression(max_iter=1000), "Logistic Regression")
train_and_evaluate(LinearSVC(), "Support Vector Machine (SVM)")


# In[ ]:




