# Decision Tree

## Process of Building a Decision Tree

For each split of a dataset $M$, do the following operations repeatedly:

- For each feature, try to split $M$ according to it and calculate the the classification error of split.
- Choose the feature with lowest classification error and split the data.
- Check whether should stop :
  - **Purity**: All data in one node belongs to one class 同一节点的所有数据的分类相同
  - **Max Depth**: Meet the limitation of depth of tree 达到了决策树的深度限制
  - **Min Samples**: There are too few data remain 当前剩余的数据个数过少 $|M|<\delta$ 

Suitable stop conditions make the tree managable and prevents overfitting.

## Pros and Cons

**Pros**:

- Handle large datasets 容易处理较大的数据集
- handle mixed predictors (continuous, discrete, qualitative) 容易混合不同的预测器（模型）
- High robustness: 
  - Can ignore redundant variables 可以忽略冗余的变量
  - Can easily handle missing data 可以处理缺失的数据
- Easy to interpret if small 规模较小的时候容易进行解释

**Cons**:

- Prediction preformance is poor 预测表现能力弱
- Does not generalize well 泛化性较弱
- Large trees are hard to interpret 规模较大的时候可解释性较差