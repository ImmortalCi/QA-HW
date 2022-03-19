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