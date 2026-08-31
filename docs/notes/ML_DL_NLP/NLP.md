---
title: NLP
tags:
  - NLP
---

# NLP

## 一、自然语言处理概念

> NLP - Natural Language Processing
>
> 作用：让计算机程序能够读懂、理解和生成人类语言的技术
>
> 应用领域：
>
> 涵盖了语音识别、语音合成、自然语言理解、机器翻译、文本分类和情感分析等多个方面
>
> •自然语言理解NLU – Natural Language Understanding
>
> •自然语言生成NLG - Natural Language Generation

## 二、文本预处理

### 1. 文本预处理

#### 1.1 概念

> 文本语料在输送给模型前一般需要一系列的预处理工作, 才能符合模型输入的要求

#### 1.2 作用

> 指导模型超参数的选择 、提升模型的评估指标

#### 1.3 主要环节

- 文本处理的基本方法 - 分词,词性标注,命名实体识别
- 文本张量表示方法 - one-hot编码,Word2vec,Word Embedding
- 文本语料的数据分析 - 标签数量分布,句子长度分布,词频统计与关键词词云
- 文本特征处理 - 添加n-gram特征,文本长度规范
- 数据增强方法 - 回译数据增强法

### 2. 文本处理的基本方法

#### 2.1 分词

> 将连续的字序列按照一定的规范重新组合成词序列的过程
>
> 工具:流行中文分词工具jieba
>
> ​	分词模式:精确模式,全模式,搜索引擎模式
>
> ​	支持中文繁体分词
>
> ​	支持用户自定义词典

- 精确模式

> 精准模式（cut_all =False） - 适合文本分析
>
> API:
>
> jieba.cut(sentence) - 得到生成器
>
> jieba.lcut(sentence) - 得到列表

- 全模式

> 全模式（cut_all =True） - 尽可能切分，词全但不重语义
>
> API:
>
> jieba.cut(sentence, cut_all=True) - 得到生成器
>
> jieba.lcut(sentence, cut_all=True) - 得到列表

- 搜索模式

> 搜索模式 - 在精确模式上在分割 - 多用于搜索引擎
>
> API：
>
> jieba.cut_for_search(sentence) - 得到生成器
>
> jieba.lcut_for_search(sentence) - 得到列表

~~~python
import jieba

sentence = '天青色等烟雨，而我在等你，炊烟袅袅升起，隔江千万里'


# 精确模式
def test01():
    result = jieba.cut(sentence)  # 优点：生成器不占用内存，用一个生成一个
    # 方法一：列表推导式 - 得到列表
    result_list1 = [item for item in result]
    print(result_list1)
    # 方法二：lcut - 得到列表
    result_list2 = jieba.lcut(sentence)
    print(result_list2)


# 全模式
def test02():
    result = jieba.cut(sentence, cut_all=True)  # 优点：生成器不占用内存，用一个生成一个
    # 方法一：列表推导式 - 得到列表
    result_list1 = [item for item in result]
    print(result_list1)
    # 方法二：lcut - 得到列表
    result_list2 = jieba.lcut(sentence, cut_all=True)
    print(result_list2)


# 搜索模式 - 在精确模式上在分割 - 多用于搜索引擎
def test03():
    # 方法一：cut_for_search
    result = jieba.cut_for_search(sentence)
    result_list1 = [item for item in result]
    print(result_list1)
    # 方法二：lcut_for_search
    result_list2 = jieba.lcut_for_search(sentence)
    print(result_list2)


if __name__ == '__main__':
    test01()
    test02()
    test03()
~~~

- 支持中文繁体分词

~~~python
import jieba

sentence = '歲月悠悠，光陰似箭，回首來時路，多少風雨多少晴。世間繁華如流水，唯有初心不可忘。願你我都能在茫茫人海中，找到屬於自己的方向，不負韶華，不枉此生。'

# 1. 精准模式
result1 = jieba.cut(sentence, cut_all=False)
# 2. 全模式
result2 = jieba.cut(sentence, cut_all=True)
# 3. 搜索模式
result3 = jieba.cut_for_search(sentence)

for word in result1:
    print(word)
~~~

- 支持用户自定义词典

> 1. 准备字典
> 2. 定义用户指定词（词 词频 词性）
> 3. 加载字典

~~~python
import jieba

sentence = '天青色等烟雨，而我在等你，炊烟袅袅升起，隔江千万里，在瓶底书汉隶仿前朝的飘逸，就当我为遇见你伏笔'
# 未使用用户字典
result = jieba.lcut(sentence)
print(result)

# ---------加载用户字典-----------------
# 1. 准备字典
# 2. 定义用户指定词（词 词频 词性）
# 3. 加载字典
jieba.load_userdict('./userdict.txt')
# ---------加载用户字典-----------------

# 使用用户字典
result = jieba.lcut(sentence)
print(result)
~~~

- 词性对照表

| 词性标签 | 解释         |
| :------- | :----------- |
| -a       | 形容词       |
| -ad      | 副形词       |
| -ag      | 形容词性语素 |
| -an      | 名形词       |
| -b       | 区别词       |
| -c       | 连词         |
| -d       | 副词         |
| -df      |              |
| -dg      | 副语素       |
| -e       | 叹词         |
| -f       | 方位词       |
| -g       | 语素         |
| -h       | 前接成分     |
| -i       | 成语         |
| -I       | 习用语       |
| -j       | 简称略称     |
| -k       | 后接成分     |
| -m       | 数词         |
| -mg      |              |
| -mq      | 数量词       |
| -n       | 名词         |
| -ng      | 名词性语素   |
| -nr      | 人名         |
| -nrfg    |              |
| -nrt     |              |
| -ns      | 地名         |
| -nt      | 机构团体名   |
| -nz      | 其他专名     |
| -o       | 拟声词       |
| -p       | 介词         |
| -q       | 量词         |
| -r       | 代词         |
| -rg      | 代词性语素   |
| -rz      | 指示代词     |
| -s       | 处所词       |
| -t       | 时间词       |
| -tg      | 时语素       |
| -u       | 助词         |
| -ud      | 结构助词得   |
| -ug      | 时态助词     |
| -uj      | 结构助词的   |
| -ul      | 时态助词了   |
| -uv      | 结构助词地   |
| -uz      | 时态助词着   |
| -v       | 动词         |
| -vd      | 副动词       |
| -vg      | 动词性语素   |
| -vi      | 不及物动词   |
| -vn      | 名动调       |
| -vq      |              |
| -x       | 非语素词     |
| -y       | 语气词       |
| -z       | 状态词       |

#### 2.2 命名实体识别NER - Named Entity Recognition

> 识别出一段文本中可能存在的命名实体(将人名, 地名, 机构名等专有名词统称命名实体)

#### 2.3 词性标注POS - Part-Of-Speech tagging

> API:
>
> from jieba import posseg as pseg
>
> pseg.cut(sentence)
>
> 将词与词性组成一个新的元组，以列表的形式返回

~~~python
from jieba import posseg as pseg

sentence = '天青色等烟雨，而我在等你，炊烟袅袅升起，隔江千万里'
result = pseg.cut(sentence)
result_list = [item for item in result]
print(result_list)
~~~

### 3. 文本张量表示方法

> 概念：将一段文本使用张量进行表示这个过程就是文本张量表示，词表示成向量叫词向量，那么一句话构成词向量矩阵
>
> 作用：将文本表示成张量（矩阵）形式，方便输入到计算机程序中进行解析

#### 3.1 one-hot编码

> One-hot编码（One-Hot Encoding），也叫稀疏词向量表示
>
> 在One-hot编码中，对于一个具有n个不同类别的分类变量，将其表示为一个n维的向量,其中只有一个维度的值为1（代表该样本属于这个类别），其他维度的值均为0。
>
> 优点：操作简单，容易理解
>
> 缺点：完全割裂了词与词之间的联系；大语料集下，每个向量的长度过大，占据大量内存
>
> ​    属于稀疏词向量表示

~~~python
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow
import joblib

# 1. 准备语料
cabs = {'关羽', '张飞', '赵云', '马超', '黄忠'}


def train_tokenizer():
    # 2. 创建Tokenizer对象
    tokenizer = tensorflow.keras.preprocessing.text.Tokenizer()
    # 通过tokenizer词映射器进行训练
    tokenizer.fit_on_texts(cabs)
    print(tokenizer.word_index)
    # 保存
    joblib.dump(tokenizer, 'tokenizer.pkl')


# 3. 加载词映射器查看one-hot编码
def use_tokenizer():
    # 加载
    tokenizer = joblib.load('tokenizer.pkl')
    for cab in cabs:
        one_hot_list = [0] * len(tokenizer.word_index)
        index = tokenizer.word_index[cab] - 1
        one_hot_list[index] = 1
        print(cab, one_hot_list)


if __name__ == '__main__':
    train_tokenizer()
    use_tokenizer()
    
"""结果：
{'张飞': 1, '赵云': 2, '黄忠': 3, '关羽': 4, '马超': 5}
张飞 [1, 0, 0, 0, 0]
赵云 [0, 1, 0, 0, 0]
黄忠 [0, 0, 1, 0, 0]
关羽 [0, 0, 0, 1, 0]
马超 [0, 0, 0, 0, 1]
"""
    
~~~

#### 3.2 Word2Vec模型

> 概念：是一种将单词转换为词向量的自然语言处理技术，是利用深度学习网络来探索单词和单词之间的语义关系，用深度学习的网络权重参数表示词向量，是在无监督的语料上构建了一个有监督的任务
>
> 方式：
>
> - CBOW（Continuous Bag of Words）方式训练词向量
> - Skip-gram方式训练词向量

##### 3.2.1 CBOW与Skip-gram的区别

> CBOW 基于两侧预测中间 
>
> 1. 选择滑动窗口3/5/7等，制作样本，用两侧预测中间
> 2. 前向计算：假若用a和c来预测b，把a送入隐藏层得到隐藏层输出1，再把b送入隐藏层得到隐藏层输出2，两个结果**加和求平均**。再送给输出层得到b的预测值
> 3. 损失函数：预测值b和真实值b有损失，计算损失
> 4. 反向传播，更新模型权重参数
> 5. 经过一轮一轮的训练模型可以学习单词和单词的特征。用权重参数当做单词词向量
> 6. 单词词向量的获取：矩阵参数与单词的one-hot编码相乘可获取词向量
>
> Skip-gram 基于中间预测两端
>
> 1. 选择滑动窗口3/5/7等，制作样本，用中间预测两边
> 2. 前向计算：假若用b来预测a和c，把b送入隐藏层得到隐藏层输出，再把b的隐藏层输出送给输出层，分别得到a的预测值和c的预测值
> 3. 损失函数：预测值a和真实值a有损失loss1，预测值c和真实值c有损失loss2，计算平均损失
> 4. 反向传播，更新模型权重参数
> 5. 经过一轮一轮的训练模型可以学习单词和单词的特征。用权重参数当做单词词向量
> 6. 单词词向量的获取：矩阵参数与单词的one-hot编码相乘可获取词向量
>
> 比较：
>
> 1. CBOW适合训练数据较大的情况，相对来说原理简单，尤其是高频词的情况下，它的计算效率高
> 2. Skip-gram对于低频词的处理更为出色，尤其在大规模数据集中，但相对计算效率较低
> 3. CBOW将多个上下文单词汇总起来，能够快速生成词向量，但可能会丧失一些细节信息
> 4. Skip-gram明确地建模了单词的上下文信息，因此在复杂的语义关系中表现更好，但训练速度相对较慢

##### 3.2.2 word2vec中使用fasttext训练词向量

> 步骤：
>
> fasttext词向量训练(**静态**-先训练后使用)
>
> 1. 获取训练数据：http://mattmahoney.net/dc/enwik9.zip；清除XML/HTML格式的内容
>
> 2. 直接训练(使用默认超参）supervised
>
> 3. 修改超参训练unsupervised
>
>    3.1 改模式 model = skipgram(默认) / cbow
>    3.2 改学习率 lr = 0.01
>    3.3 改维度 dim = 50
>    3.4 改迭代次数 epoch = 10
>    3.5 改线程数 thread = 10
>
> 4. 模型的保存 model.save_model()
>
> 5. 模型加载 model.load_model()

~~~python
import fasttext


def dem01():
    # 1. 使用fasttext训练一个模型
    model = fasttext.train_supervised('./data/wh02aa')  # 默认使用skipgram
    # 2. 保存训练好的模型
    model.save_model('./model/wh02aa.model')


def dem02():
    # 1. 加载模型
    # model = fasttext.load_model('./model/wh02aa.model')
    # 测试修改超参后的模型
    model = fasttext.load_model('./model/wh02aa_new.model')
    # 2. 使用模型查看词的维度
    vector = model.get_word_vector('china')
    print(vector)
    print(vector.shape)
    # 3. 查看当前词周边词
    result = model.get_nearest_neighbors('china')
    print(result)


def dem03():
    # 1. 修改训练的超参数（修改使用cbow）
    model = fasttext.train_unsupervised('./data/wh02aa', 'cbow', dim=50, lr=0.001, epoch=10, thread=4)
    # 2. 保存模型
    model.save_model('./model/wh02aa_new.model')


if __name__ == '__main__':
    # dem01()
    dem02()
    # dem03()
~~~

#### 3.3 Word Embedding

> 定义：通过一定的方式将词汇映射到指定维度（一般是更高维度）的空间
>
> 广义：所有密集词汇向量的表示方法
>
> 狭义：深度神经网络中嵌入一个层
>
> API：embed = torch.nn.Embedding(词数量，词维度)
>
> 注：权重参数会参与更新，是**动态**的

### 4. 案例 - 文本张量tensorboard可视化

> 1. 切词
> 2. 将词列表转换为[词表]
> 3. 搭建embedding层
> 4. 通过词表的索引建立词向量
> 5. 数据可视化

~~~python
import jieba
import tensorflow
import torch.nn
from torch.utils.tensorboard import SummaryWriter

sentence1 = '天青色等烟雨，而我在等你，月色袅袅升起，隔江千万里'
sentence2 = '在瓶底书汉隶仿前朝的飘逸'

# 1. 切词
word_list = []
sentences = [sentence1, sentence2]
for sentence in sentences:
    words = jieba.lcut(sentence)
    word_list.append(words)
print(word_list)

# 2. 将词列表转换为[词表]
tokenizer = tensorflow.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(word_list)
print(tokenizer.word_index)

# 3. 搭建embedding层
embed = torch.nn.Embedding(len(tokenizer.word_index), 128)

# 4. 通过词表的索引建立词向量
for word in tokenizer.word_index:  # word_index中索引是从1开始的
    print(word, embed(torch.tensor([tokenizer.word_index[word] - 1])).shape)

# 5.1 数据可视化(创建)
# 输出结果存放目录，一会可视化需要在对应目录启动
summary = SummaryWriter(log_dir='./logs')  # 默认存放在runs目录下  路径不能有中文
# 参数：1. 模型的权重 2. 展示的label标签
summary.add_embedding(embed.weight.data, metadata=tokenizer.word_index.keys())
summary.close()

# 5.2 数据可视化(启动前端工程)
# 1 找到创建后的目录
# 2 启动命令：tensorboard --logdir=logs --host 0.0.0.0
# 3 首次启动需要等待1分钟左右，访问给出地址(无法展示)
# 4 将访间地址改为：http://localhost:6006
~~~

### 5. 文本数据分析

> 作用：文本数据分析能够有效帮助我们理解数据语料, 快速检查出语料可能存在的问题, 指导模型训练过程中一些超参数的选择

#### 5.1 标签数量分布

> 训练深度学习模型比如分类问题，一般需要将正负样本比例维持在1:1左右，不符合1:1比例，需进行**数据增强或删减**

~~~python
import pandas as pd
import matplotlib.pyplot as plt

# 设置引擎
plt.rcParams['backend'] = 'TkAgg'
# 设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置负号问题
plt.rcParams['axes.unicode_minus'] = False

train_data = pd.read_csv('./data/train.tsv', sep='\t')
test_data = pd.read_csv('./data/dev.tsv', sep='\t')

# 标签为1的数量
label1 = sum(train_data['label'].values)
# label1 = len(train_data['label'].values[train_data['label'].values == 1])
# 标签为0的数量
label0 = len(train_data) - label1
# label0 = len(train_data['label'].values[train_data['label'].values == 0])

# 设置538风格
plt.style.use('fivethirtyeight')
plt.bar(['训练集标签为1的数量', '训练集标签为0的数量'], [label1, label0])
plt.show()

# 测试集
label_test1 = sum(test_data['label'].values)
label_test0 = len(test_data) - label_test1

plt.bar(['测试集标签为1的数量', '测试集标签为0的数量'], [label_test1, label_test0])
plt.show()
~~~

#### 5.2 句子长度分布

##### 5.2.1 句子长度分布柱状图和密度曲线图

> 若模型对输入的数据长度有要求，可以对句子进行截断或补齐操作；(规范长度)起到关键的指导作用
>
> **map(要处理的函数，处理的数据）** 执行完成后会返回新的处理后的数据

~~~python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置引擎
plt.rcParams['backend'] = 'TkAgg'
# 设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置负号问题
plt.rcParams['axes.unicode_minus'] = False

train_data = pd.read_csv('./data/train.tsv', sep='\t')
test_data = pd.read_csv('./data/dev.tsv', sep='\t')

# 数据添加一列：[句子的长度]
# map(要处理的函数，处理的数据） 执行完成后会返回新的处理后的数据
sentence_length = list(map(lambda sentence: len(sentence), train_data['sentence']))
train_data['sentence_length'] = sentence_length

# 绘制 - 训练
# 设置538风格
plt.style.use('fivethirtyeight')
# 柱状图
sns.countplot(x='sentence_length', data=train_data)
# 清空当前坐标轴（Axes）上 X 轴的所有刻度标签和刻度线
plt.xticks([])
plt.show()
# 密度曲线图（kde设置曲线走向）
sns.histplot(x='sentence_length', data=train_data, kde=True)
plt.xticks([])
plt.show()
~~~

##### 5.2.2 正负样本长度散点图

> 通过查看正负样本长度散点图, 可有效定位异常点的出现位置, 帮助我们更准确进行人工语料审查

~~~python
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('TkAgg')
train_data = pd.read_csv('./data/train.tsv', sep='\t')

train_data['sentence_length'] = train_data['sentence'].str.len()

# 散点图
plt.style.use('fivethirtyeight')  # 设置538风格
sns.stripplot(y='sentence_length', x='label', data=train_data)
plt.show()
~~~

#### 5.3 词频统计与关键词词云

##### 5.3.1 高频词云绘制

> 1. 训练集数据
> 2. 筛选出满足的数据
>      正样本 -> 布尔索引实现
>      样本 -> 拼接成字符串
>      字符串 -> pseg切词
>      过滤词性为形容词的词
>      得到了满足条件的词列表
> 3. 初始化Wc对象
>      对象.generate(''.join(词列表))
> 4. 使用plt绘制imshow(wc)

~~~python
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from jieba import posseg as pseg
from wordcloud import WordCloud

matplotlib.use('TkAgg')
# 1. 训练集
train_data = pd.read_csv('./data/train.tsv', sep='\t')
# 2. 正样本
train_data_pos = train_data[train_data['label'] == 1]
# 3. 样本中的句子取出来
train_pos_sentences = train_data_pos['sentence'].values
text = ' '.join(train_pos_sentences)
# 4. 词性标注
cut_words_list = pseg.cut(text)
# 形容词列表
words_list = []
for word in cut_words_list:
    # 筛选出所有的形容词
    if word.flag == 'a':
        words_list.append(word.word)
# 5. 词云绘制
wc = WordCloud(font_path='./data/LXGWWenKaiMono-Light.ttf', background_color='white', width=1000, height=800,
               margin=10, ).generate(
    ' '.join(words_list))
plt.figure()
plt.imshow(wc)
plt.axis('off')
plt.show()
~~~

##### 5.3.2 chain函数

> *的作用：列表前面使用，作用是拆包
>
> chain函数的作用：将多个列表合并成1个列表（直接合并:List1+list2）

~~~python
from itertools import chain

# *
print(*[[1, 2, 3], [4, 5, 6]])

# chain函数
result = chain(*[[1, 2, 3], [4, 5, 6]])  # 把data最外层括号拆掉，然后合并成一个新列表
print(result)  # 返回一个itertools（可迭代）对象
# for i in result:
#     print(i)
print(list(result))
~~~

##### 5.3.3 zip函数

> zip压缩函数，将相同维度的值进行压缩
>
> zip返回一个itertools（可迭代）对象

~~~python
data = [1, 2, 3, 4, 5]

# 需求：需要[(1,2),(2,3),(3,4),(4,5)]
print(list(zip(data, data[1:])))

# 需求：需要[(1,2,3),(2,3,4),(3,4,5)]
print(list(zip(data, data[1:], data[2:])))
~~~

### 6. 文本特征处理

> 语料添加具有普适性的文本特征，让模型更有效的处理数据，提高模型性能指标

#### 6.1 n-gram特征

> 一个词一个词看句子，有些信息可能无法理解
> 需要从多个词去理解句子，需要添加新的特征（从n个词的维度理解语句，就叫做n-gram）
> 1-gram，也叫uni-gram
> 2-gram，也叫bi-gram
> 3-gram，也叫tri-gram

~~~python
data = [1, 2, 1, 2, 3, 4, 1, 2, 5]
# 目标：将data提取2-gram
# 2-gram：[(1,2),(2,3),(3,4),(4,5)]
n_gram = 2
# 压缩后可能会有重复，需要去重
res = zip(*[data[i:] for i in range(n_gram)])
print('未去重：', list(res))
res = zip(*[data[i:] for i in range(n_gram)])
print('去重后：', set(res))

# 3-gram：[(1,2,3),(2,3,4),(3,4,5)]
n_gram = 3
res = set(zip(*[data[i:] for i in range(n_gram)]))
print(res)
~~~

#### 6.2 文本长度处理

> 模型训练时，语料长度是固定的，但数据长度参差不齐
> 处理：截断或补齐
> API：sequence.pad_sequences
>
> padding：'pre'/'post'   truncating：'pre'/'post'
>
> 默认前面补全(padding='pre')
> 默认前面截断(truncating='pre') - 把前面的截掉

~~~python
from keras.preprocessing import sequence

data = [[1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
res1 = sequence.pad_sequences(data, maxlen=8)
print('前面补全，前面截断', res1)

res2 = sequence.pad_sequences(data, maxlen=8, padding='post')
print('后面补全，前面截断', res2)

res3 = sequence.pad_sequences(data, maxlen=8, truncating='post')
print('前面补全，后面截断', res3)

res4 = sequence.pad_sequences(data, maxlen=8, padding='post', truncating='post')
print('后面补全，后面截断', res4)
~~~

### 7. 文本数据增强

> 回译数据增强法
>
> 一般基于google/百度/获取其他翻译接口，将文本数据**翻译**成另外一种语言(一般选择小语种),之后**再翻译**回原语言；即可认为得到与与原语料同标签的**新语料**。
>
> 优势：操作简便, 获得新语料质量高
>
> 问题：1. 短文本语料易重复    2. 多次翻译语料易失真

## 三、RNN及其变体

### 1. RNN（Recurrent Neural Network）

> 以序列数据为输入, 通过网络内部的结构设计有效捕捉序列之间的关系特征, 一般也是以序列形式进行输出
>
> 输入有两个：数据端输入，上一时间步的隐藏层输入
>
> 输出有两个：数据端输出，本时间步的隐藏层输出

> 分类：
>
> 从输入输出的角度：
>
> - N vs N – RNN    输入N个序列，输出N个序列                写诗 写对联 固定场景表达
>
> - N vs 1 – RNN    输入N个序列，输出1个的值                 情感分类 意图识别
>
> - 1 vs N – RNN    输入1个序列，输出N个的值                 看图说话
>
> - N vs M – RNN   输入N个序列，输出M个序列                 翻译、语音转换；文本生成、摘要
>
> 从内部结构角度：
>
> 传统RNN、LSTM、Bi-LSTM、GRU、Bi-GRU

### 2. 传统RNN

> 优点：1. 内部结构简单，对计算资源要求低     2. 在短序列任务上性能和效果都表现优异
>
> 缺点：长序列文本特征提取效果差，过长的序列易导致梯度消失或爆炸
>
> 参数：
>
> 创建RNN时：1. input_sizeRNN层输入的维度（词向量）2. hidden_sizeRNN输出的维度   3. num_layers隐藏层数量
>
> 输入x：1. seq_len(句子长度)  2. batch-size(批次大小）  3. input_size(词向量维度)
>
> 隐藏层h：1.隐藏层层数 2.batch-size(批次大小) 3.hidden_size输出向量维度
>
> 输出output：1. seq_len(句子长度)  2. batch-size(批次大小）  3. hidden_size(输出向量维度)
>
> 内部计算公式：
> $$
> 学术界：h_t = \tanh(W_t[X_t, h_{t-1}] + b_t)
> $$
>
> $$
> pytorch框架：h_t = \tanh(W_{ih} \cdot X_t + b_{ih} + W_{hh} \cdot h_{t-1} + b_{hh})
> $$
>
> 

| 符号             | 含义                                                         |
| ---------------- | ------------------------------------------------------------ |
| $h_t$            | 当前时刻 t 的隐藏状态                                        |
| $h_{t-1}$        | 上一时刻 t-1 的隐藏状态                                      |
| $X_t$            | 当前时刻 t 的输入向量                                        |
| $W_{ih}$         | 输入到隐藏的权重矩阵（Input-to-Hidden）                      |
| $W_{hh}$         | 隐藏到隐藏的权重矩阵（Hidden-to-Hidden），即循环权重         |
| $b_{ih}$         | 输入部分的偏置向量                                           |
| $b_{hh}$         | 隐藏状态部分的偏置向量                                       |
| $tanh$           | 双曲正切激活函数，输出范围 (-1, 1)                           |
| $[X_t, h_{t-1}]$ | 将输入 $X_t$和上一时刻隐藏状态$ h_{t-1}$ 拼接成一个更长的向量 |
| $W_t$            | 权重矩阵，用于线性变换拼接后的向量                           |
| $b_t$            | 偏置向量                                                     |

~~~python
import torch

# 1. 创建RNN层
# 参数（RNN层输入的维度（词向量），RNN输出的维度（自定义），隐藏层数量）
rnn = torch.nn.RNN(input_size=2, hidden_size=3, num_layers=2)

# 2. 准备x与h0
# 参数：1. seq_len(句子长度)  2. batch-size(批次大小）  3. input_size(词向量维度)
x = torch.randn(2, 4, 2)
# 参数说明：1.隐藏层层数 2.batch-size批次大小 3.hidden_size输出向量维度
h0 = torch.randn(2, 4, 3)

# 3. 进入RNN层，得到h1和output
# 注1：先给输入，后给状态
# 注2: 先拿结果，后拿状态
# output参数说明：1. seq_len(句子长度)  2. batch-size(批次大小）  3. hidden_size(输出向量维度)
# h1参数与h0相同
output, h1 = rnn(x, h0)
print(output.shape, h1.shape)
print(output)
print(h1)

# 注1 ：当隐藏层个数配置成1时  output 和 hn 输出是一样的
# 注2 ：如果隐藏层配置n个，则output的最后一个和最后一个隐藏层输出是一样的
~~~

### 3. LSTM

![LSTM](.\assets\LSTM.png)

>优点：LSTM的门结构能够有效**减缓**长序列问题中可能出现的梯度消失或爆炸
>
>缺点：由于内部结构相对较复杂，因此训练效率在同等算力下较传统RNN低很多
>
>API：
>
>lstm = torch.nn.LSTM(input_size=2, hidden_size=3, num_layers=1, bidirectional=True)
>
>参数说明：bidirectional双向LSTM（默认单向） -> 开启后h和c的隐藏层数量翻倍
>
>细胞状态c：1.隐藏层层数 2.batch-size批次大小 3.hidden_size输出向量维度

~~~python
import torch

# 1. 创建LSTM层
# 参数（LSTM层输入的维度（词向量），LSTM输出的维度（自定义），隐藏层数量）
# bidirectional双向LSTM（默认单向） -> 开启后h和c的隐藏层数量翻倍
lstm = torch.nn.LSTM(input_size=2, hidden_size=3, num_layers=1, bidirectional=True)

# 2. 准备输入的数据
# 参数：1. seq_len(句子长度)  2. batch-size(批次大小）  3. input_size(词向量维度)
x = torch.randn(2, 3, 2)
# 参数说明：1.隐藏层层数 2.batch-size批次大小 3.hidden_size输出向量维度
h0 = torch.randn(2, 3, 3)
# 参数说明（与h相同）：1.隐藏层层数 2.batch-size批次大小 3.hidden_size输出向量维度
c0 = torch.randn(2, 3, 3)

# 3. 进入LSTM层，得到h1,c1和output
# output参数说明：1. seq_len(句子长度)  2. batch-size(批次大小）  3. hidden_size(输出向量维度))
# h1参数与h0相同
output, (h1, c1) = lstm(x, (h0, c0))
print(output.shape, h1.shape, c1.shape)
~~~

### 4. GRU

![GRU](.\assets\GRU.png)

> 优点：在捕捉长序列语义关联时, 能有效**抑制**梯度消失或爆炸, 效果都优于传统RNN
>
> ​		计算复杂度相比LSTM要小
>
> 缺点：不能完全解决梯度消失问题
>
> ​		 不可并行计算
>
> API：gru = torch.nn.GRU(input_size=2, hidden_size=3, num_layers=2, bidirectional=True)
>
> 参数说明：bidirectional双向GRU默认单向） -> 开启后h的隐藏层数量翻倍

~~~python
import torch

# 1. 创建GRU层
# 参数（GRU层输入的维度（词向量），GRU输出的维度（自定义），隐藏层数量）
# bidirectional双向GRU默认单向） -> 开启后h的隐藏层数量翻倍
gru = torch.nn.GRU(input_size=2, hidden_size=3, num_layers=2, bidirectional=True)

# 2. 准备x与h0
# 参数：1. seq_len(句子长度)  2. batch-size(批次大小）  3. input_size(词向量维度)
x = torch.randn(2, 4, 2)
# 参数说明：1.隐藏层层数 2.batch-size批次大小 3.hidden_size输出向量维度
h0 = torch.randn(4, 4, 3)

# 3. 进入GRU层，得到h1和output
# output参数说明：1. seq_len(句子长度)  2. batch-size(批次大小）  3. hidden_size(输出向量维度))
# h1参数与h0相同
output, h1 = gru(x, h0)
print(output.shape, h1.shape)
~~~

## 四、迁移学习

### 1. fasttext介绍

> 简介：是自然语言处理（NLP）任务的开源工具包，由Facebook AI Research（FAIR）开发
>
> 作用：进行文本分类，训练词向量
>
> 优势：在保持较高精度的情况下, 快速的进行训练和预测是fasttext的最大优势
>
> 优势原因：
>
> - fasttext工具包中内含的fasttext模型，**模型结构简单**
> - 使用fasttext模型训练词向量时，使用**层次softmax**结构, 来提升超多类别下的模型性能
> - **采用负采样（negative sampling）**，每次训练仅仅更新一小部分的权重, 降低梯度下降过程中的计算量
> - 采用n-gram特征提取文本特征以弥补模型缺陷提升精度

### 2. fasttext模型架构

#### 2.1 三层架构![fasttext三层架构](.\assets\fasttext三层架构.png)

#### 2.2 层次softmax

> 对比：
>
> 传统softmax：计算词表中**每一个词**的得分，再分别计算每个词的概率
>
> 层次softmax：不直接计算词表中所有词的概率，而是将词表中的词作为**叶子节点**构建一棵二叉树（通常是哈夫曼树）。预测时，模型只需要从根节点出发，在每个分叉路口做一次二分类（走左边还是右边），直到走到目标词所在的叶子节点。最终概率是**路径上所有二分类概率的乘积**
>
> 霍夫曼树的作用：用来构建带权值最小的二叉树

![霍夫曼树](.\assets\霍夫曼树.png)

##### 2.2.1 构建霍夫曼树

> 假设有四个Label分别为: A~D, 统计其在语料库出现的频数为A(5次), B(9次), C(7次), D(3次)
>
> 假设有n个权值, 则构造出的哈夫曼树有n个叶子节点. n个权值分别设为 w1、w2、…、wn, 则哈夫曼树的构造规则为:
>
> 步骤1: 将w1、w2、…, wn看成是有n 棵树的森林(每棵树仅有一个节点);
>
> 步骤2: 在森林中选出两个根节点的权值最小的树合并, 作为一颗新树的左、右子树, 且新树的根节点权值为其左、右子树根节点权值之和;
>
> 步骤3: 从森林中删除选取的两棵树, 并将新树加入森林;
>
> 步骤4: 重复2-3步骤, 直到森林只有一颗树为止, 该树就是所求的哈夫曼树.

##### 2.2.2 训练霍夫曼树

![训练霍夫曼树](.\assets\训练霍夫曼树.png)

#### 2.3 负采样

> 负采样每次让一个训练样本仅更新一小部分的权重, 这样就会降低梯度下降过程中的计算量
>
> 以前：1个正例9999个反例，1个正例更新9999个反例更新
> 现在：1个正例9999个反例，1个正例更新10个反例更新

### 3. fasttext文本分类

> 文本分类：是将文档（例如电子邮件，帖子，文本消息，产品评论等）分配给一个或多个类别
>
> 训练文本分类模型需要是有监督学习，需要标签
>
> - 文本分类种类：
>
>   二分类：文本被分类为两个类别中的一个类别
>
>   单标签多分类：文本可被分成多个类别中的一个类别（即被打上某一个标签)
>
>   多标签多分类：文本可被分成多个类别中的多个类别（即被打上多个标签)
>
> - 步骤：
>
>   •获取数据
>
>   •训练集与验证集的划分
>
>   •训练模型
>
>   •使用模型进行预测并评估
>
>   •模型调优
>
>   •模型保存与重加载
>
> - 调优方法：
>
>   •重新处理数据
>
>   •增加训练轮数:
>
>   •调整学习率:
>
>   •增加n-gram特征:
>
>   •修改损失计算方式:
>
>   •自动超参数调优:
>
>   •实际生产中多标签多分类问题的损失计算方式
>
> - API：
>
>   训练：fasttext.train_supervised（）
>
>   ​	参数：epoch（训练轮数），lr（学习率），wordNgrams（n-gram），loss（损失计算方法：'ns' - 负采样，'hs' - 层次softmax，'softmax' - 单标签，'ova' - one-versus-all 多标签），autotuneValidationFile='验证数据' 自动超参数调优 + autotuneDuration 时间（单样本预测时要设置k：要几个标签，和阈值（threshold）为几（标签概率大于几））
>
>   预测：model.predict（单样本预测），model.test（验证测试集）
>
>   保存：model.save_model
>
>   加载：fasttext.load_model

~~~python
import fasttext


# 直接训练
def dm00():
    # 1. 准备模型（直接使用fasttext训练模型）
    model = fasttext.train_supervised('./data/cooking_train.txt')
    # 2. 模型预测
    # 单样本预测
    result1 = model.predict('How do I fix a cast iron pot that was heated empty for hours?')
    print(f'result1：{result1}')
    # 验证集测试
    result2 = model.test('./data/cooking_valid.txt')
    print(f'result2：{result2}')


# 基于直接训练，清洗原数据
def dm01():
    # 1. 准备模型
    model = fasttext.train_supervised('./data/cooking.pre.train')
    # 2. 模型预测
    # 单样本预测
    result1 = model.predict('How do I fix a cast iron pot that was heated empty for hours?')
    print(f'result1：{result1}')
    # 验证集测试
    result2 = model.test('./data/cooking.pre.valid')
    print(f'result2：{result2}')


# 基于dm01，增加训练轮数
def dm02():
    # 1. 准备模型
    model = fasttext.train_supervised('./data/cooking.pre.train', epoch=30)
    # 2. 模型预测
    # 单样本预测
    result1 = model.predict('How do I fix a cast iron pot that was heated empty for hours?')
    print(f'result1：{result1}')
    # 验证集测试
    result2 = model.test('./data/cooking.pre.valid')
    print(f'result2：{result2}')


# 基于dm02，调整学习率
def dm03():
    # 1. 准备模型
    model = fasttext.train_supervised('./data/cooking.pre.train', epoch=30, lr=0.35)
    # 2. 模型预测
    # 单样本预测
    result1 = model.predict('How do I fix a cast iron pot that was heated empty for hours?')
    print(f'result1：{result1}')
    # 验证集测试
    result2 = model.test('./data/cooking.pre.valid')
    print(f'result2：{result2}')


# 基于dm03，调整n-gram
def dm04():
    # 1. 准备模型
    model = fasttext.train_supervised('./data/cooking.pre.train', epoch=30, lr=0.35, wordNgrams=2)
    # 2. 模型预测
    # 单样本预测
    result1 = model.predict('How do I fix a cast iron pot that was heated empty for hours?')
    print(f'result1：{result1}')
    # 验证集测试
    result2 = model.test('./data/cooking.pre.valid')
    print(f'result2：{result2}')


# 基于dm04，修改损失计算方法(默认softmax，可以改为'ns' - 负采样，'hs' - 层次softmax)
def dm05():
    # 1. 准备模型
    model = fasttext.train_supervised('./data/cooking.pre.train', epoch=30, lr=0.35, wordNgrams=2, loss='ns')
    # 2. 模型预测
    # 单样本预测
    result1 = model.predict('How do I fix a cast iron pot that was heated empty for hours?')
    print(f'result1：{result1}')
    # 验证集测试
    result2 = model.test('./data/cooking.pre.valid')
    print(f'result2：{result2}')


# 自动超参数调优
def dm06():
    # 1. 准备模型
    # autotuneValidationFile='验证数据' 自动超参数调优
    # autotuneDuration 时间
    model = fasttext.train_supervised('./data/cooking.pre.train', autotuneValidationFile='./data/cooking.pre.valid',
                                      autotuneDuration=60 * 3)
    # 2. 模型预测
    # 单样本预测
    result1 = model.predict('How do I fix a cast iron pot that was heated empty for hours?')
    print(f'result1：{result1}')
    # 验证集测试
    result2 = model.test('./data/cooking.pre.valid')
    print(f'result2：{result2}')


# 修改损失方式，适配多标签分类
# loss='softmax' - 单标签（最终只选出一个概率最高的标签）
# loss='ova' - one-versus-all 多标签（选出多个符合的标签）
def dm07():
    # 1. 准备模型
    model = fasttext.train_supervised('./data/cooking.pre.train', epoch=30, lr=0.2, wordNgrams=2, loss='ova')
    # 2. 模型预测
    # 单样本预测
    # 要两个标签k，阈值（threshold）为0.5（标签概率大于0.5）
    result1 = model.predict('How do I fix a cast iron pot that was heated empty for hours?', k=2, threshold=0.5)
    print(f'result1：{result1}')
    # 阈值为0.5的标签全都要
    result2 = model.predict('regulation and balancing of readymade packed mayonnaise and other sauces', k=-1,
                            threshold=0.5)
    print(f'result2：{result2}')
    # 验证集测试
    result3 = model.test('./data/cooking.pre.valid')
    print(f'result3：{result3}')


# 保存模型
# model.save_model
def dm08():
    model = fasttext.train_supervised('./data/cooking.pre.train', epoch=30, lr=0.2, wordNgrams=2, loss='ova')
    # 保存模型
    model.save_model('./model/cooking.model')


# 加载模型
# fasttext.load_model
def dm09():
    model = fasttext.load_model('./model/cooking.model')
    # 模型预测
    # 2. 模型预测
    # 单样本预测
    # 要两个标签k，阈值（threshold）为0.5（标签概率大于0.5）
    result1 = model.predict('How do I fix a cast iron pot that was heated empty for hours?', k=2, threshold=0.5)
    print(f'result1：{result1}')
    # 阈值为0.5的标签全都要
    result2 = model.predict('regulation and balancing of readymade packed mayonnaise and other sauces', k=-1,
                            threshold=0.5)
    print(f'result2：{result2}')
    # 验证集测试
    result3 = model.test('./data/cooking.pre.valid')
    print(f'result3：{result3}')


# 加载模型
if __name__ == '__main__':
    # dm00()
    # dm01()
    # dm02()
    # dm03()
    # dm04()
    # dm05()
    # dm06()
    # dm07()
    dm08()
    dm09()
~~~

### 4. 迁移学习

> 概念：利用在一个任务上学到的知识来改善在另一个相关任务上的性能
>
> 形式：
>
> ​	•特征迁移： 将在一个任务上训练的模型的一部分（如词嵌入或者更高层的特征）应用于另一个任务。
>
> ​	•模型迁移： 将在一个任务上训练的模型应用于另一个任务，通常会对源模型进行微调以适应新任务。
>
> ​	•知识迁移： 将从源任务中学到的知识（例如，词嵌入、模型参数等）应用于目标任务。
>
> 相关术语：预训练模型，微调
>
> 两种迁移方式：
>
> ​	开箱即用：直接使用预训练模型
>
> ​	微调预训练模型：继承预训练模型+微调参数
>
> 预训练模型的分类：
>
> ​	•自回归语言模型（Auto Regressive Language Model AR）：生成 - GPT
>
> ​	•自编码语言模型（Auto Encoder Language Model AE）：理解 - bert
>
> ​	•Seq2Seq编码和解码混合类型的语言模型：生成+理解：T5

## 五、Transformers库使用

> Transformers是开源的、基于 transformer 模型结构的，提供预训练语言库

### 1. 三层应用结构

- 管道方式（Pipline）：高度集成的极简使用方式，只需要几行代码即可实现一个NLP任务
- 自动模型方式（AutoMode）：可载入并使用BERTology系列模型
- 具体模型方式（SpecificModel）：在使用时，需要明确指定具体的模型，并按照每个BERTology系列模型中的特定参数进行调用，该方式相对复杂，但具有较高的灵活度

> - 文本分类任务：'sentiment-analysis'
>   -模型：chinese_sentiment
> - 特征提取任务：'feature-extraction'
>   -模型：bert-base-chinese
> - 完形填空任务：'fill-mask'
>   -模型：chinese-bert-wwm
> - 阅读理解任务：'question-answering'
>   -模型：chinese_pretrain_mrc_roberta_wwm_ext_large
> - 文本摘要任务：'summarization'
>   -模型：distilbart-cnn-12-6
> - 命名实体识别(NER)任务：'ner'
>   -模型：roberta-base-finetuned-cluener2020-chinese

### 2. 管道模式

~~~python
import numpy as np
from transformers import pipeline

# transformers的使用方式为3种：管道、自动模型、指定模型
model_path = r"D:\itheima\AI\PretrainedModel"


# 管道模式
# 文本分类任务：'sentiment-analysis'
# -模型：chinese_sentiment
def dm01():
    # 1. 实例化pipeline
    model = pipeline(task='sentiment-analysis', model=model_path + r'\chinese_sentiment')
    # 2. 模型预测
    result = model('我非常喜欢这个模型')
    # 3. 查看模型预测结果
    print(result)


# 特征提取任务：'feature-extraction'
# -模型：bert-base-chinese
def dm02():
    # 1. 实例化pipeline
    model = pipeline(task='feature-extraction', model=model_path + r'\bert-base-chinese')
    # 2. 模型预测
    result = model('我非常喜欢这个模型')
    # 3. 查看模型预测结果
    print(np.array(result).shape)  # (1, 11, 768)
    # 样本 = 1
    # 词长度 = 11，为什么不是9？[cls(分类)] [句子=9] [sep(分隔符)]
    # 词向量 = 768


# 完形填空任务：'fill-mask'
# -模型：chinese-bert-wwm
def dm03():
    # 1. 实例化 pipeline
    model = pipeline(task='fill-mask', model=model_path + r'\chinese-bert-wwm')
    # 2. 模型预测
    result = model('[MASK]学习，天天向上')
    # 3. 查看模型预测结果
    print(result)


# 阅读理解任务：'question-answering'（已废弃）
# -模型：chinese_pretrain_mrc_roberta_wwm_ext_large
def dm04():
    pass


# 文本摘要任务：'summarization'（已废弃）
# -模型：distilbart-cnn-12-6
def dm05():
    pass


# 命名实体识别(NER)任务：'ner'
# -模型：roberta-base-finetuned-cluener2020-chinese
def dm06():
    # 1. 实例化 pipeline
    model = pipeline(task='ner', model=model_path + r'\roberta-base-finetuned-cluener2020-chinese')
    # 2. 模型预测
    result = model('  老潘要去  三 亚 旅游了')
    # result = model('O O O O  B  I O O O')  # B(begin)I(inside)O(outside)规则
    # [{'entity': 'B-address', 'score': np.float32(0.77222395), 'index': 5, 'word': '三', 'start': 4, 'end': 5},
    #  {'entity': 'I-address', 'score': np.float32(0.7430532), 'index': 6, 'word': '亚', 'start': 5, 'end': 6}]
    # 3. 查看模型预测结果
    print(result)
    result1 = model('阿祖收手吧，外面全是成龙')
    print(result1)


if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    dm06()
~~~

### 3. 自动模型模式

~~~python
import numpy as np
import torch
# AutoModel - 自动加载模型
# AutoConfig - 自动加载模型配置
# AutoTokenizer - 自动加载分词器
from transformers import AutoConfig, AutoModel, AutoTokenizer
# 文本分类：AutoModelForSequenceClassification
# 掩码任务：AutoModelForMaskedLM
# 阅读理解：AutoModelForQuestionAnswering
# 摘要识别：AutoModelForSeq2SeqLM
# ner：AutoModelForTokenClassification
from transformers import AutoModelForMaskedLM, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification

# transformers的使用方式为3种：管道、自动模型、指定模型
model_path = r"D:\itheima\AI\PretrainedModel"


# 自动模型模式
# 文本分类任务：'sentiment-analysis'
# -模型：chinese_sentiment
def dm01():
    # 1. 准备tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path + r'\chinese_sentiment')
    # 2. 准备模型
    model = AutoModelForSequenceClassification.from_pretrained(model_path + r'\chinese_sentiment')
    # 3. 数据转张量
    msg = '我非常喜欢这个模型'
    # 把文本转张量
    # 参数说明:1. 要编码的数据  2. 返回的张量类型(return_tensors)pt(pytorch)|tf(tensorflow)|np(numpy)
    # 3. padding填充  4. truncation截断  5. max_length最大长度
    # 新版本：padding = 'max_length'
    msg_tensor = tokenizer.encode(msg, return_tensors='pt', padding='max_length', truncation=True, max_length=15)
    print(msg_tensor)
    # 4. 模型预测
    model.eval()
    result = model(msg_tensor)
    print(f'结果是：star{torch.argmax(result.logits.squeeze()) + 1}')


# 特征提取任务：'feature-extraction'
# -模型：bert-base-chinese
def dm02():
    # 效果：输入一个文本，输出一个特征向量
    # 不带任务输出头输出
    # 带任务输出头输出
    # 1. 创建tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path + r'\bert-base-chinese')
    # 2. 创建模型
    model = AutoModel.from_pretrained(model_path + r'\bert-base-chinese')
    # 3. 文本转向量
    msgs = ['我非常喜欢这个模型', '骗你的']
    msgs_tensor = tokenizer.encode(msgs, return_tensors='pt', padding='max_length', truncation=True, max_length=8)
    # 4. 模型预测
    model.eval()
    result = model(msgs_tensor)
    print(result.last_hidden_state)
    # result.last_hidden_state - 最后的隐藏层的状态
    # result.pooler_output - 池化层的输出


# 完形填空任务：'fill-mask'
# -模型：chinese-bert-wwm
def dm03():
    # 1. 创建tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path + r'\chinese-bert-wwm')
    # 2. 创建模型
    model = AutoModelForMaskedLM.from_pretrained(model_path + r'\chinese-bert-wwm')
    # 3. 数据转张量
    msg = '[MASK]学习，天天[MASK]上'
    msg_tensor = tokenizer.encode(msg, return_tensors='pt', padding='max_length', truncation=True, max_length=15)
    # 4. 模型预测
    model.eval()
    result = model(msg_tensor)
    # print(result.logits.shape)
    # torch.Size([1, 15, 21128])  1句话，15个词，每个词的概率
    # 21128 - 不是词向量维度，而是所有的词
    print(f'结果是：{torch.argmax(result.logits.squeeze()[1])}')  # 第一个要预测的词
    print(f'结果是：{tokenizer.convert_ids_to_tokens(torch.argmax(result.logits.squeeze()[1]).item())}')
    print(f'结果是：{torch.argmax(result.logits.squeeze()[7])}')  # 第二个要预测的词
    print(f'结果是：{tokenizer.convert_ids_to_tokens(torch.argmax(result.logits.squeeze()[7]).item())}')
    # 看概率最高的前5个 - torch.topk(k=5)
    # values - 概率的值
    # indices - 概率的索引
    print(f'结果是：{torch.topk(result.logits.squeeze()[1], k=5)}')
    print(f'结果是：{tokenizer.convert_ids_to_tokens(torch.topk(result.logits.squeeze()[1], k=5).indices)}')
    print(f'结果是：{torch.topk(result.logits.squeeze()[7], k=5)}')
    print(f'结果是：{tokenizer.convert_ids_to_tokens(torch.topk(result.logits.squeeze()[7], k=5).indices)}')


# 阅读理解任务：'question-answering'（已废弃）
# -模型：chinese_pretrain_mrc_roberta_wwm_ext_large
def dm04():
    # 1. 创建tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path + r'\chinese_pretrain_mrc_roberta_wwm_ext_large')
    # 2. 创建模型
    model = AutoModelForQuestionAnswering.from_pretrained(model_path + r'\chinese_pretrain_mrc_roberta_wwm_ext_large')
    # 3. 数据转张量
    context = '老潘要去三亚旅游了'
    questions = ['老潘要去哪里了', '老潘要干什么']
    for question in questions:
        # 参数：1. 问题  2. 上下文
        question_tensor = tokenizer.encode(question, context, return_tensors='pt')
        print(question_tensor)  # [CLS] + 单个question + [SEP] + context + [SEP]
        # 4. 模型预测
        model.eval()
        result = model(question_tensor)
        # print(f'问题是：{question}，结果是：{result}')  # start_logits，end_logits
        # 获取答案的开始和结束索引
        start, end = torch.argmax(result.start_logits.squeeze()), torch.argmax(result.end_logits.squeeze())
        # 通过索引拿答案
        print(f'问题：{question}，结果是：{tokenizer.convert_ids_to_tokens(question_tensor.squeeze()[start:end + 1])}')


# 文本摘要任务：'summarization'（已废弃）
# -模型：distilbart-cnn-12-6
# 注\表示当前行未输入完，下一行继续输入
def dm05():
    # 1. 创建tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path + r'\distilbart-cnn-12-6')
    # 2. 创建模型
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path + r'\distilbart-cnn-12-6')
    # 3. 数据转张量
    text = "BERT is a transformers model pretrained on a large corpus of English data " \
           "in a self-supervised fashion. This means it was pretrained on the raw texts " \
           "only, with no humans labelling them in any way (which is why it can use lots " \
           "of publicly available data) with an automatic process to generate inputs and " \
           "Iabels from those texts. More precisely, it was pretrained with two objectives:Masked " \
           "language modeling (MLM): taking a sentence, the model randomly masks 15% of the " \
           "words in the input then run the entire masked sentence through the model and has " \
           "to predict the masked words. This is different from traditional recurrent neural " \
           "networks (RNNs) that usually see the words one after the other, or from autoregressive " \
           "models like GPT which internally mask the future tokens. It allows the model to learn " \
           "a bidirectional representation of the sentence.Next sentence prediction (NSP): the models" \
           "concatenates two masked sentences as inputs during pretraining.Sometimes they correspond to" \
           "sentences that were next to each other in the origina text, sometimes not. The model then " \
           "has to predict if the two sentences were following each other or not."
    text_tensor = tokenizer.encode(text, return_tensors='pt')
    # 4. 模型预测
    model.eval()
    # 生成摘要generate（会有特殊符号和空格）
    result = model.generate(text_tensor)
    # print(result)  # 二维
    # 解码：
    # 去除特殊符号：skip_special_tokens
    # 去除空格：clean_up_tokenization_spaces
    res = tokenizer.decode(result.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(res)


# 命名实体识别(NER)任务：'ner'
# -模型：roberta-base-finetuned-cluener2020-chinese
def dm06():
    # 0. 自动加载配置
    config = AutoConfig.from_pretrained(model_path + r'\roberta-base-finetuned-cluener2020-chinese')
    # 1. 创建tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path + r'\roberta-base-finetuned-cluener2020-chinese')
    # 2. 创建模型
    model = AutoModelForTokenClassification.from_pretrained(model_path + r'\roberta-base-finetuned-cluener2020-chinese')
    # 3. 数据转张量
    msg = '阿祖收手吧，外面全是成龙'
    msg_tensor = tokenizer.encode(msg, return_tensors='pt')
    # 4. 模型预测
    model.eval()
    result = model(msg_tensor)
    # print(result.logits.shape)  # [1, 14, 32] 一个样本，14个词，32个分类的概率
    # 文本里的每个字->['[CLS]', '阿', '祖', '收', '手', '吧', '，', '外', '面', '全', '是', '成', '龙', '[SEP]']
    input_tokens = tokenizer.convert_ids_to_tokens(msg_tensor.squeeze())
    # 将字和字的类型的概率压缩
    zip_data = zip(input_tokens, result.logits.squeeze())  # 生成器
    output = []
    # 遍历
    for token, logits in zip_data:
        # 排除特殊符号(all_special_tokens)
        if token in tokenizer.all_special_tokens:
            continue
        # 指定格式（字，类型）
        # id2label - 类型字典
        output.append((token, config.id2label[logits.argmax().item()]))
    print(output)


if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    # dm04()
    # dm05()
    dm06()
~~~

### 4. 指定模型模式

~~~python
# 指定模型和自动模型基本一摸一样，只有模型加载的API需要进行调整，其他基本不变
# 演示： 完形填空修改成指定模型
import torch
from transformers import BertTokenizer, BertForMaskedLM

# transformers的使用方式为3种：管道、自动模型、指定模型
model_path = r"D:\itheima\AI\PretrainedModel"


# 指定模型模式：
# 完形填空任务：'fill-mask'
# -模型：chinese-bert-wwm
def dm03():
    # 1. 创建tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path + r'\chinese-bert-wwm')
    # 2. 创建模型
    model = BertForMaskedLM.from_pretrained(model_path + r'\chinese-bert-wwm')
    # 3. 数据转张量
    msg = '好[MASK]学习，天天[MASK]上'
    msg_tensor = tokenizer.encode(msg, return_tensors='pt', padding='max_length', truncation=True, max_length=15)
    # 4. 模型预测
    model.eval()
    result = model(msg_tensor)
    # print(result.logits.shape)
    # torch.Size([1, 15, 21128])  1句话，15个词，每个词的概率
    # 21128 - 不是词向量维度，而是所有的词
    print(f'结果是：{torch.argmax(result.logits.squeeze()[2])}')  # 第一个要预测的词
    print(f'结果是：{tokenizer.convert_ids_to_tokens(torch.argmax(result.logits.squeeze()[2]).item())}')
    print(f'结果是：{torch.argmax(result.logits.squeeze()[8])}')  # 第二个要预测的词
    print(f'结果是：{tokenizer.convert_ids_to_tokens(torch.argmax(result.logits.squeeze()[8]).item())}')
    # 看概率最高的前5个 - torch.topk(k=5)
    # values - 概率的值
    # indices - 概率的索引
    print(f'结果是：{torch.topk(result.logits.squeeze()[2], k=5)}')
    print(f'结果是：{tokenizer.convert_ids_to_tokens(torch.topk(result.logits.squeeze()[2], k=5).indices)}')
    print(f'结果是：{torch.topk(result.logits.squeeze()[8], k=5)}')
    print(f'结果是：{tokenizer.convert_ids_to_tokens(torch.topk(result.logits.squeeze()[8], k=5).indices)}')


if __name__ == '__main__':
    dm03()
~~~

