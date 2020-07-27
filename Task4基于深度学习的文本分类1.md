# 文本的表示方法

## FastText、Word2Vec和Bert

# FastText

FastText是一种典型的深度学习词向量的表示方法，它非常简单通过Embedding层将单词映射到稠密空间，然后将句子中所有的单词在Embedding空间中进行平均，进而完成分类操作。

在https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext中 下载fasttext-0.9.2-cp37-cp37m-win_amd64.whl，并用pip进行安装


```python
import numpy as np
import pandas as pd
import warnings
import fasttext
from sklearn.metrics import f1_score
```


```python
train_df = pd.read_csv('train_set.csv', sep='\t')
test_df = pd.read_csv('test_a.csv', sep='\t')
```


```python
data_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
train_count = train_df.shape[0]
test_count = test_df.shape[0]
val_count = int(train_df.shape[0] * 0.20)
data_df['label_ft'] = '__label__' + data_df['label'].astype(str)
data_df[['text','label_ft']].iloc[:-(val_count+test_count)].to_csv('train_set.csv', index=None, header=None, sep='\t')
```


```python
model = fasttext.train_supervised('train_set.csv', lr=1.0, wordNgrams=2, verbose=2, minCount=1, epoch=25, loss="hs")
```


```python
val_pred = [model.predict(x)[0][0].split('__')[-1] for x in data_df.iloc[-(val_count+test_count):-test_count]['text']]
test_pred = [model.predict(x)[0][0].split('__')[-1] for x in data_df.iloc[-test_count:]['text']]
print(f1_score(data_df['label'].values[-(val_count+test_count):-test_count].astype(str), val_pred, average='macro'))
```

    0.913626631959226
    


```python

```
