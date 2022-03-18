# 实验数据记录

## 初筛模型：

### fast_text 评估指标（3.18更新）：

使用test_fordata.py脚本跑出，训练权重文件 save/stage1_3.15

<img width="199" alt="image" src="https://user-images.githubusercontent.com/74886609/159014757-35a6cd08-affd-44ee-8920-728a4b6e9fb0.png">

### BM25 评估指标：

使用elastic search跑的测试集结果(3.17更新)

使用scripts/get_bm25_data.py脚本跑出，无训练权重文件

<img width="285" alt="image" src="https://user-images.githubusercontent.com/74886609/158788064-59cfc781-14aa-4ac0-9770-b7402267fad7.png">


## 精排模型：

### fast_text作为初筛下的指标（3.18更新）：

使用scripts/test_ranking.py脚本跑出，训练权重文件 save/stage2_fasttext_3.15

<img width="211" alt="image" src="https://user-images.githubusercontent.com/74886609/159015089-e5928b4f-9562-4281-ba91-523398301a5d.png">



### BM25作为初筛下的指标（3.17更新）：

使用scripts/test_ranking_bm25.py脚本跑出，训练权重文件 save/stage2_bm25

<img width="267" alt="image" src="https://user-images.githubusercontent.com/74886609/158792735-06275f07-d939-4152-a4d6-bcdb3aa0bec6.png">

