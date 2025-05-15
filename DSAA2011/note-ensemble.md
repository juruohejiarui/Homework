# Ensemble

Methods to improve the performance of wak learners, it shifts responsibility from 1 weak learner to an "ensemble" of such weak learners. A set of weak learners are combined to form a strong learner with better performance than any of them individually. 将多个较弱的模型结合获取较好的效果

## Basic Concepts

Assume that we have $N$ **independent** base learners, the error rate of each base learner is $\epsilon$, then the error rate of the ensemble model is :

$$
\sum_{i=1}^{\left\lfloor\frac{N}{2}\right\rfloor}
	\binom{N}{k} (1-\epsilon)^{N-k}\epsilon^k \leq \exp \left(-\frac{1}{2}N(1-2\epsilon)^2\right)
$$

If $\epsilon<0.5$, the error rates drop down when $N$ increases.

This can be understood using Hoeffding's Inequality.

Tips: Applying Hoeffding's Inequality with :

- $X_i\sim B(1, 1-\epsilon), \mathbb{E}\left[X_i\right]=1-\epsilon, a=0, b=1$

## Bagging

### Step1: Sampling

The the original data set $\mathcal{D}$ , generate $N$ (e.g. $N=10$ ) bootstrap samples $\mathcal{D}_1, \dots, \mathcal{D}_N$ , where $\forall i=1,\dots, N, |\mathcal{D}_i|=|\mathcal{D}|$, by sampling with replacement 通过有放回的随机抽样来获取 $N$ 个大小和原数据集相同的子数据集

by the formula : $\lim_{n\rightarrow +\infty}\left(1-\frac{1}{n}\right)^n=\frac{1}{e}\simeq 0.368$, the probability of that one sample is not in any $\mathcal{D}_i$ is nearly $63.2\%$ . The sample that is not chosen is called **out-of-bag** sample.

### Step2: Training

Train one model on each bootstrap sample. Each model learns slightly different patterns due to sample variation. Then we get $N$ weak leaarners. 

### Step3: Aggregation

- For **classification**: use majority voting;
- For **regression**: average the predictions.

### Out-of-bag Samples (OOB)

ununsed samples are used in :

- calculate out-of-bag estimate (e.g. cross-validation)
- Hyper-paramter tuning
- Early-stopping to prevent overfitting

### Issues

The expected error of a model is :

$$
\mathbb{E}\left[\left(f-\hat f\right)^2\right]=\left(f-\mathbb{E}\left[\hat f\right]\right)^2+\mathbb{E}\left[\left(\hat f-\mathbb{E}\left[\hat f\right]\right)^2\right]
$$

since that $\mathbb{E}\left[\hat f\right] = \mathbb{E}\left[\hat {f_i}\right]$ where $\hat {f_i}$ is the model trained with $\mathcal{D}_i$, the bias of bagged models is the same as that of individual models. Bagging 之后的偏置和Bagging前相同。Each model is identically distributed (i.d. but not i.i.d, not independent). Bagging trees are still corelated. Correlated models may not reduce variance.


Bagging may reduce variance by :

- calculate predictions multiple times
- average the predictions
- then make certain estimations

## Random Forests

Aim to decorelate the trees generated for bagging. The base learner is decision tree.

This algorithm introudce diversity by:

1. randomness in bootstrapping
2. randomness in feature selection

Assume that we have $n$ bootstrapped splits. For each split in $n$ trees, consider only $k$ features from the full feature set $m$. where $k<m$ .

hyper-paramters of random forests:

- the number of predictors to randomly select at each split 随机森林中的决策树中可以使用的预测器的种类数
- the number of tree in the ensemble 总的模型数量/Bootstrap Sample数据集个数
- the minimum leaf node size 决策树的最少叶节点个数
- number of features used in one tree $k$. 

These hyper-paramters can be tuned through OOB

## Boosting

