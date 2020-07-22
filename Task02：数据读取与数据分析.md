# 任务一：

> 假设字符3750，字符900和字符648是句子的标点符号，请分析赛题每篇新闻平均由多少个句子构成?

```python
train_df['res'] = train_df['text'].apply(lambda x:sum([x.count('3750'),x.count('900'),x.count('648')]))
train_df['res'].mean()
# result:79.80237
```

# 任务二：

> 统计每类新闻中出现次数对多的字符

```python
from collections import Counter
for i in range(0,14):
    tmp = train_df.loc[train_df['label'] == i,]
    all_lines = ' '.join(list(tmp['text'])) 
    word_count = Counter(all_lines.split(" "))
    word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)
    
for i in range(0,14):
    print(word_count[i][0],word_count[i][1])
"""
3750 33796
648 26867
900 11263
4939 9651
669 8925
6122 8321
4893 7605
3864 6241
4811 6069
1465 5684
3800 5525
7399 5297
3070 4816
6040 4725

"""
```

