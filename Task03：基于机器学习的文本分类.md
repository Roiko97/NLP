```python
from sklearn.feature_extraction.text import CountVectorizer
```


```python
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
vectorizer = CountVectorizer()
vectorizer.fit_transform(corpus).toarray()
```




    array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
           [0, 2, 0, 1, 0, 1, 1, 0, 1],
           [1, 0, 0, 1, 1, 0, 1, 1, 1],
           [0, 1, 1, 1, 0, 0, 1, 0, 1]], dtype=int64)




```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.linear_model import RidgeClassifier 
from sklearn.metrics import f1_score
 
```


```python
train_df = pd.read_csv('train_set.csv', sep='\t', nrows=15000)
vectorizer = CountVectorizer(max_features=3000) 
train_test = vectorizer.fit_transform(train_df['text'])
clf = RidgeClassifier() 
clf.fit(train_test[:10000], train_df['label'].values[:10000])
 
val_pred = clf.predict(train_test[10000:]) 
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro')) 
```

    0.7416952793751392
    


```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import RidgeClassifier 
from sklearn.metrics import f1_score

```


```python
train_df = pd.read_csv('train_set.csv', sep='\t', nrows=15000)
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000) 
train_test = tfidf.fit_transform(train_df['text'])
clf = RidgeClassifier() 
clf.fit(train_test[:10000], train_df['label'].values[:10000])
val_pred = clf.predict(train_test[10000:]) 
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
```

    0.8721598830546126
    


```python

```
