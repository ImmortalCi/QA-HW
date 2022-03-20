# 实验数据记录

## 初筛模型：

### fast_text 评估指标（3.18更新）：

使用 scripts/test_for_test_data.py脚本跑出，训练权重文件 save/stage1_3.15

![](images/fast_text_stage1.jpg)

### BM25 评估指标：

使用elastic search跑的测试集结果(3.17更新)

使用scripts/get_bm25_data.py脚本跑出，无训练权重文件

![](images/BM25_stage1.jpg)

## 精排模型：

### fast_text作为初筛下的指标（3.18更新）：

使用scripts/test_ranking.py脚本跑出，训练权重文件 save/stage2_fasttext_3.15

![](images/fast_text_stage2.jpg)

### BM25作为初筛下的指标（3.17更新）：

使用scripts/test_ranking_bm25.py脚本跑出，训练权重文件 save/stage2_bm25

![](images/BM25_stage2.jpg)



| Stage | Model         | R@1   | R@3   | R@5   | R@10  | R@20  |
| ----- | ------------- | ----- | ----- | ----- | ----- | ----- |
| 召回  | BM25          | 58.81 | 71.95 | 74.90 | 77.35 | 79.07 |
| 召回  | FastText      | 62.02 | 77.98 | 81.26 | 84.21 | 86.42 |
| 精排  | BM25+BERT     | 63.31 | 75.26 | 77.05 | 78.34 | 79.07 |
| 精排  | FastText+BERT | 66.09 | 81.35 | 83.68 | 85.66 | 86.42 |

