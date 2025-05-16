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

Boosting is to take an ensemble of simple models $\left\{T_h\right\}_{h\in H}$ and additively combine them into a single, more complex model. Each model $T_h$ might be a poor fit for the data, but a linear combination of the ensemble can be expressive. 训练若干个小模型，然后通过类似“相加”的方式，将模型组合起来。最终模型为效果较差小模型的线性组合。

$$
T=\sum_{h\in H}\lambda_h T_h
$$

The problem is to decide the weights $\{\lambda_h\}$

**Pros**:

- High accuracy
- Flexibility
- AdaBoost and Gradient Boost can rank feature importance.

**Cons**:

- Overfitting: 
  - AdaBoost: Sensitive to noisy data and outliers AdaBoost对噪声敏感
  - Gradient Boosting: Too many trees opr poor tuning can overfit 太多模型或者错误的微调
- Computational Cost:
  - Both: Sequential training-slower than parallel methods like bagging 模型训练有先后顺序
  - Gradient Boosting: Especially resource-heavy with many iterations 当迭代次数过多的时候计算强度较大


### Gradient Boosting

Process:

1. Fit a simple model $T=T^{(0)}$ on the training set $\{(x_1, y_1),\dots, (x_n, y_n)\}$
2. Compute the residuals $\{(r_1,\dots, r_n)\}$ for model $T$ , where $r_i=y_i-T(x_i)$ .
3. Fit a simple $T^{(1)}$ to the current residual, i.e. $\{(x_1, r_1),\dots,(x_n, r_n)\}$
4. Build a new model $T\leftarrow T+\lambda T^{(1)}$
5. Compute the updated residuals, $r_n\leftarrow r_n-\lambda T^{(1)}(x_n)$
6. Repeated step 2~5 until the stopping condition met.

Translate into Chinese:

1. 首先计算出基底模型
2. 计算和答案之间的差值
3. 使用相同结构的小模型拟合差值
4. 将小模型按 $\lambda$ 的权重加入到基底模型中
5. 使用小模型按 $\lambda$ 的权重削减差值
6. 重复Step 2到Step 5

In practice, the objective functions are complicated, and analytically finding the stationary point is intractable. And we use the gradient descents to update residual. i.e. use the simple model $T^{(1)}$ to fit $\frac{\partial L(y_i,T(x_i))}{\partial T(x_i)}$ 在实际应用中，目标函数（损失函数）较为复杂，而人们常用损失函数对预测值的求导的梯度来更新差值。

choice of learning rate:

- constant, should be tuned by cross-validation
- variable
  - around the optimum, where the gradient is small, $\lambda$ is small
  - otherwise, $\lambda$ should be large
  
**Tips**: boosting can overfit. Gradient Boosting can handle any loss function.

### AdaBoost

AdaBoost works for classification task and the output $y_i\in\{1,-1\}$ of each sample $(x_i,y_i)\in \mathcal{D}$

Process:

- Initalize equal weights $\frac{1}{N}$ for all training samples, i.e. $\forall i=1,\dots,n, w_i=\frac{1}{N}$
- For each iteration 
  - Train a weak learner $T^{(t)}$ on weighted data $\mathcal{D}^{(t)}=\{(x_i,w_iy_i)\}$ and get the error rate $\epsilon^{(t)}$
  - If $\epsilon^{(t)}>0.5$ then "continue" iteration without updating weights and models
  - Compute the classifier weight $\lambda^{(t)}=\frac{1}{2}\ln\left(\frac{1-\epsilon^{(t)}}{\epsilon^{(t)}}\right)$
  - Update weights using $w_i\leftarrow \frac{1}{Z}w_i\cdot\exp\left(-\lambda^{(t)}y_iT^{(t)}(x_i)\right) \text{ where } Z=\sum_{i=1}^n \exp\left(-\lambda^{(t)}y_iT^{(t)}(x_i)\right)$
- Combine all weak classifiers $T=\sum_t \lambda^{(t)}T^{(t)}$

When $\epsilon^{(t)}$ is small, $\lambda^{(t)}$ is high. $\lambda^{(t)}$ should minimize exponential loss of the learner $T^{(t)}$.

$$
\begin{aligned}
&L_{\mathrm{exp}}=\sum_{i=1}^n \exp\left(-y_i\hat y_i\right) \\
&\text{If apply }\lambda^{(t)}\text{ then it becomes} \\
&L_{\mathrm{exp}}=\sum_{i=1}^n \exp\left(-y_i\lambda^{(t)} \hat y_i\right) \\
&\text{then we let} \frac{\partial L_{\mathrm{exp}}}{\partial \lambda^{(t)}} = 0 \\
&\Rightarrow \lambda^{(t)}=\frac{1}{2}\ln \left(\frac{1-\epsilon^{(t)}}{\epsilon^{(t)}}\right)
\end{aligned}
$$

since that $w_i^{(t+1)}\leftarrow \frac{1}{Z}w_i^{t}\cdot \exp\left(-\lambda^{(t)}y_iT^{(t)}(x_i)\right)$ , 

$$
\begin{aligned}
&\forall \left(x^{(t+1)}_i,y^{(t+1)}_i\right)\in\mathcal{D}^{(t+1)}, \\
&y^{(t+1)}_i \propto y^{(t)}_i \exp\left(-y_i\lambda^{(t)}T^{(t)}(x_i)\right)\\
\Rightarrow&y^{(t+1)}_i \propto y_i\exp\left(-y_i\left(\sum_{i=1}^t \lambda^{(t)}T^{(t)}(x_i)\right)\right)
	& \text{obtained by mathematical induction}
\end{aligned}
$$

Then we train $T^{(t+1)}$ to minimize the loss of $T+T^{(t+1)}$ where $T=\sum_{i=1}^t \lambda^{(t)}T^{(t)}$:

$$
\begin{aligned}
\text{since } &\mathbb{E}_{(x,y)\in\mathcal{D}}\left[\exp\left(-y\left(T(x)+T^{(t+1)}(x)\right)\right)\right] \\
=&\mathbb{E}_{(x,y)\in\mathcal{D}}\left[\exp\left(-yT(x)\right)\exp\left(-yT^{(t+1)}(x)\right)\right] \\
\approx&\mathbb{E}_{(x,y)\in\mathcal{D}}\left[\exp\left(-yT(x)\right)\left(1-yT^{(t+1)}(x)+\frac{1}{2}y^2T^{(t+1)}(x)^2\right)\right] & \text{obtained by taylor expansion} \\
=&\mathbb{E}_{(x,y)\in\mathcal{D}}\left[\exp\left(-yT(x)\right)\left(\frac{3}{2}-yT^{(t+1)}(x)\right)\right] \\
\text{then } T^{(t+1)*}&=\arg\max_{T^{(t+1)}}\left\{\mathbb{E}_{(x,y)\in\mathcal{D}}\left[\exp\left(-yT(x)\right)yT^{(t+1)}(x)\right]\right\} \\
=&\arg\max_{T^{(t+1)}}\left\{\mathbb{E}_{(x,y)\in\mathcal{D}}\left[
		\frac{\exp\left(-yT(x)\right)}{\mathbb{E}_{(x,y)\in\mathcal{D}}\left[\exp\left(-yT(x)\right)\right]}
		yT^{(t+1)}(x)
	\right]
\right\} \\
=&\color{red}\arg\max_{T^{(t+1)}}\left\{\mathbb{E}_{(x,y)\in\mathcal{D}^{(t+1)}}\left[
		yT^{(t+1)}(x)
	\right]
\right\}
\end{aligned}
$$

## Feature Selection

Choose a subset of relevant features from dataset. To Improve accuracy, reduce overfitting, speed up training.

增强准确性，减少过拟合，提高训练速度

### Filter

Rank features before modeling using stats.

- Numeric: Correlation; Categorical: chi-squared. mutual information
- Using prior knowledge

**Pros**: fast, simple, scalable

**Cons**: Misses feature interactions

### Wrapper

Test feature subsets by training a model. 

Operations: Adding, Removing, Recursive elimination.

**Pros**: Model-specific, considers interactions

**Cons**: Slow, overfit

### Embedding

Selection during model training.

Lasso (shrinks wak features), trees (split on key features)

**Pros**: Efficient, tailored to model （效率高，量身定制）

**Cons**: Depends on model choice