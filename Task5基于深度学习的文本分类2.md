# 基于深度学习的文本分类2

对于每一条输入文本，我们选取-一个上下文窗口和一一个中心词，并基于这个中心词去预测窗口里其他词出现的概率。因此，word2vec模型可以方便地从新增语料中学习到新增词的向量表达，是一种高效的在线学习算法(online learning)。

word2vec的主要思路:通过单词和上下文彼此预测，对应的两个算法分别为:
1. Skip-grams (SG):预测上下文

2. Continuous Bag of Words (CBOW):预测目标单词

另外提出两种更加高效的训练方法:

1. Hierarchical softmax
2. Negative sampling

## Skip-grams 的原理

Word2Vec模型中，主要有Skip-Gram和CBOW两种模型，从直观上理解，Skip-Gram是给定input word来预测上下文。而CBOW是给定上下文，来预测input word。

Word2Vec模型实际上分为了两个部分，第一部分为建立模型，第二部分是通过模型获取嵌入词向量。

![image-20200731195342472](C:\Users\cheng\AppData\Roaming\Typora\typora-user-images\image-20200731195342472.png)

Word2Vec的整个建模过程实际上与自编码器（auto-encoder）的思想很相似，即先基于训练数据构建一个神经网络，当这个模型训练好以后，我们并不会用这个训练好的模型处理新的任务，我们真正需要的是这个模型通过训练数据所学得的参数，例如隐层的权重矩阵——后面我们将会看到这些权重在Word2Vec中实际上就是我们试图去学习的“word vectors”。

## Hierarchical Softmax 

输入：权值为(w1,w2,…wn)的n个节点

输出：对应的霍夫曼树

1. 将(w1,w2,…wn)看做是有n棵树的森林，每个树仅有一个节点
2. 在森林中选择根节点权值最小的两棵树进行合并，得到一个新的树，这两颗树分布作为新树的左右子树。新树的根节点权重为左右子树的根节点权重之和
3. 将之前的根节点权值最小的两棵树从森林删除，并把新树加入森林
4. 重复步骤 2 和 3 直到森林里只有一棵树为止

## TextCNN

TextCNN利用CNN（卷积神经网络）进行文本特征抽取，不同大小的卷积核分别抽取n-gram特征，卷积计算出的特征图经过MaxPooling保留最大的特征值，然后将拼接成一个向量作为文本的表示。

这里我们基于TextCNN原始论文的设定，分别采用了100个大小为2,3,4的卷积核，最后得到的文本向量大小为100*3=300维。

![image-20200731195619769](C:\Users\cheng\AppData\Roaming\Typora\typora-user-images\image-20200731195619769.png)



## TextRNN

TextRNN利用RNN（循环神经网络）进行文本特征抽取，由于文本本身是一种序列，而LSTM天然适合建模序列数据。TextRNN将句子中每个词的词向量依次输入到双向双层LSTM，分别将两个方向最后一个有效位置的隐藏层拼接成一个向量作为文本的表示。

![image-20200731195701776](C:\Users\cheng\AppData\Roaming\Typora\typora-user-images\image-20200731195701776.png)