# 数据库调研

## [AmazonQA](https://github.com/amazonqa/amazonqa)

训练集文件名：train-qar.jsonl

### asin

不太清楚具体含义

### 类别

### 问题文本

### 问题类型

### 评论片段

### 回答

这里的回答不是唯一的，可能有多个回答

- 回答文本
- 回答类型
- 回答质量

  使用包含两个整数的列表来表示
  第一个整数：认为这个答案有用的数量
  第二个整数：所有对这个答案的评论数
  
### 是否可回答

### question_id

## [MS_MARCO](https://github.com/microsoft/MSMARCO-Question-Answering)

目前只看了一下V2.1的训练集每一部分作为一个大整体来存贮的
并不是像GitHub主页上的example一样挨个存储的。

### Answers

### Passages

### query

### query_id

### query_type

### wellFormedAnswers

## [Quora duplicate questions](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)

### query1

### query2

### is_duplicate

前面两个query是否相似

## [AFQMC(蚂蚁金融语义相似度数据集)](https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip)

     数据量：训练集（34334）验证集（4316）测试集（3861）
     例子：
     {"sentence1": "双十一花呗提额在哪", "sentence2": "里可以提花呗额度", "label": "0"}
     每一条数据有三个属性，从前往后分别是 句子1，句子2，句子相似度标签。其中label标签，1 表示sentence1和sentence2的含义类似，0表示两个句子的含义不同。

### sentence1

### sentence2

### label

## [LCQMC](https://www.luge.ai/#/luge/dataDetail?id=14)

哈工大的通用领域问题匹配数据集
训练集大小238766，开发集大小8802，测试集大小12500


### 和AFQMC结构相同

## [cMedQA2](https://github.com/zhangsheng93/cMedQA2)

医学在线论坛的数据，包含10万个问题，及对应的约20万个回答。
有分好的train_candidate，三元组形式，但是存储的是各自的ID。

### 没有展开看结构

## [WikiQA](https://www.microsoft.com/en-us/download/confirmation.aspx?id=52419)

数据集中包含了 3,047 个问题
label来表示正样本或者负样本，共29058条数据


### QuestionID

### Question

### DocumentID

### DocumentTitle

### SentenceID

### Sentence

### Label

## [hotpotQA](https://github.com/hotpotqa/hotpot)

对支持事实具有强大的监督，以实现更可解释的问答系统。

### id

### question

### answer

### supporting facts

### context

## [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/)

和阅读理解类似的东西；很经典的数据集

### qas

- question
- id
- answers

### context

## [TREC](http://trec-liveqa.org/)

TREC Live QA Track Website 打不开
可能时间过于久远，没找到数据集地址


