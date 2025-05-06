# Mathematical Tools

## Basic Functions

- **Logistic Function / Sigmoid Function**: $f(x)=\frac{1}{1 + \exp(-x)}$
- $\mathbb{I}\{\text{statement}\}=\begin{cases}1 & \text{statement is true} \\ 0 & \text{otherwise}\end{cases}$
- $[x]^+=\max\{0,x\}$ 


## Probability and Statistic

### Guassian Noise 高斯噪声

$$
f(x|\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{x^2}{2\sigma^2}\right)
$$

### Bayes's rule 贝叶斯
$\mathcal{X}, \mathcal{Y}$ : event space

$$
p(y|x)=\frac{p(x|y)p(y)}{\sum_{y'\in \mathcal{Y}}p(x|y')p(y')}
$$

### Expectation 期望
- Discrete case 离散情况：

$$
\mathbb{E} Y= \mathbb{E}[g(X)]=\sum_{y\in \mathcal{Y}} y\cdot p_Y(y)=\sum_{x\in \mathcal{X}} g(x)p_X(x)
$$

- Continuous case 连续情况：
where $f$ is the probability density funtion 概率密度函数

$$
\mathbb{E} Y=\mathbb{E}[g(X)]=\int_{\mathbb{R}} yf_Y(y)\mathrm{d}y=\int_{\mathbb{R}}g(x)f_X(x)\mathrm{d}x
$$

### Maximum Likelihood Estimation (MLE) 最大似然估计
Assume we get a probability function $p_X(x; \theta)$ with unconfirmed parameters, $\theta$ . Additionally, there is a data set $x=\{x_1,x_2,\dots,x_n\}$ , then the MLE of $\theta$ is :

$$
\begin{aligned}
\hat{\theta}_{\mathrm{ML}} &= \arg\max_{\theta} p_{X}(x_1, x_2, \dots, x_n; \theta) \\
&=\arg\max_{\theta} \log p_{X}(x_1, x_2, \dots, x_n; \theta)
\end{aligned}
$$

if $x_1, x_2, \dots, x_n$ are independent, then:

$$
\begin{aligned}
\hat{\theta}_{\mathrm{ML}} &= \arg\max_{\theta} \prod_{i=1}^n p_{X}(x_i; \theta) \\
&= \arg\max_{\theta} \sum_{i=1}^n \log p_{X}(x_i; \theta)
\end{aligned}
$$

MLE can be biased （最大似然估计不一定是无偏的）

#### Consistency 
$\hat\theta_{\mathrm{ML}}\xrightarrow{p} \theta$ as $n\rightarrow \infty$ . We say $X_n \xrightarrow{p} X$ as $n\rightarrow \infty$ if 

$$
\lim_{n\rightarrow}\mathrm{Pr}(\|X_n-X\|>\epsilon) = 0, \forall \epsilon\in \mathbb{R}
$$

#### Asymptotic Normality 渐进正态
we have $\sqrt{n} \left(\hat\theta_{\mathrm{ML}} - \theta\right)\xrightarrow{d} \mathcal{N}(0, I(\theta)^{-1})$ as $n\rightarrow \infty$, where $I(\theta)$ is fisher information of $\theta$ (larger information $\Longrightarrow$ smaller variance)

We say $X_n\xrightarrow{d}X$ as $n\rightarrow \infty$ if 

$$
\lim_{n\rightarrow \infty}F_{X_n}(x)=F(x)~\forall x \text{  where }f\text{ is continuous}
$$

#### Model Sensitivity

MLE is sensitive to model assumptions, and incorrect assumptions can lead to biased or inconsistent estimates. MLE 对模型的选择比较敏感，错误的模型选择会导致偏差'

### Standard Normal Distribution 标准正态分布

This is normal distribution with $\mu=0, \sigma^2=1$ .

Probability density function (PDF): $f(x)=\frac{1}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right)$

Cumulative distributionm function (CDF): $\Phi(x)=\int_{-\infty}^x f(x) \mathrm{d} x$

### Central Limit Theorem (CLT) 中心极限定理
Suppose $X_1, X_2, X_3, \dots$ is a sequence of i.i.d random variables with $\mathbb{E}[X_i]=\mu$ and $\mathrm{var}[X_i]=\sigma<\infty$ . then as $n\rightarrow\infty$ , the distribution of $\sqrt{n}(\overline{X}_n-\mu)$ converges to $\mathcal{N}(0, \sigma^2)$ . 

This implies that, when $\sigma>0$, the CDF of $\sqrt{n}(\overline{X}_n-\mu)$ converge pointwise to the CDF of standard normal CDF, which is :

$$
\lim_{n\rightarrow \infty} \mathrm{Pr}(\sqrt{n}(\overline{X}_n-\mu)\leq z)=\lim_{n\rightarrow \infty} \mathrm{Pr}\left(\frac{\sqrt{n}(\overline{X}_n-\mu)}{\sigma}\leq \frac{z}{\sigma}\right)=\Phi\left(\frac{z}{\sigma}\right)
$$

## Linear Algebra

### Symbols

- For a matrix $\mathrm{X}\in \mathbb{R}^{m\times n}$ , $\mathrm{X}^{(k)}$ represents the $k$-th colutmn of $\mathrm{X}$ , which is a vector in $n$-dimensional vector space.

- $\mathrm{I}_n$ , an $n\times n$ matrix where the element on the main diagonal is $1$, otherwise $0$. 

### Rouch´ e-Capelli Theorem
For sytem $\mathrm{X}\mathbf{w}=\mathbf{y}$ where $\mathrm{X}\in \mathbb{R}^{m\times n}, \mathbf{w}\in \mathbb{R}^{n}, \mathbf{y}\in \mathbb{R}^m$ , where we need to find a solution of variable $\mathbf{w}$. 

let $\~{\mathrm{X}}=\begin{bmatrix}\mathrm{X}&\mathbf{y}\end{bmatrix}$ be an argumented matrix.

- this system admits a **unique** solution $\Longleftrightarrow$ $\mathrm{rank}(\mathrm{X})=\mathrm{rank}(\~{\mathrm{X}})=n$
- this system has **no** solution $\Longleftrightarrow$ $\mathrm{rank}(\mathrm{X})<\mathrm{rank}(\~{\mathrm{X}})$
- this system has **infinity** many solution $\Longleftrightarrow$ $\mathrm{rank}(\mathrm{X})=\mathrm{rank}(\~{\mathrm{X}})<n$

### Woodbury Formula
$$
(\mathrm{I}-\mathrm{U}\mathrm{V})^{-1}=\mathrm{I}-\mathrm{U}(\mathrm{I}+\mathrm{V}\mathrm{U})^{-1}\mathrm{V}
$$

### Least Square Solution

For a linearly system $\mathrm{X}\mathbf{w}=\mathbf{y}$ , where $\mathrm{X}\in \mathbb{R}^{m\times n}, \mathbf{w}\in \mathbb{R}^n, \mathbf{y}\in\mathbb{R}^m$, the least square solution is :

$$
\~{\mathbf{w}}=\left(\mathrm{X}^\top\mathrm{X}\right)^{-1}\mathrm{X}^\top\mathbf{y}
$$

### Unamed Theorems

- $\mathrm{rank}(\mathrm{A})=\mathrm{rank}(\mathrm{A^\top A})$
- if $\mathrm{A}\in \mathbb{R}^{n\times n}$ is positive or negative definite, then $\mathrm{A}$ is invertible.

### Norm of Vector

$$
\lVert\mathbf{w}\rVert _p=\left(\sum_{i=1}^p |\mathbf{w}|^p\right)^{\frac{1}{p}}
$$

when $p=0$: not actually a norm, $\lVert\mathbf{w}\rVert _0=\sum_{i=1}^d \mathbb{I}(w_i\ne 0)$

when $p=\infty$: $\lVert\mathbf{w}\rVert _{\inf}=\max_i|w_i|$

## Evaluation of Model

### Error Metric

**Confusion Matrix**: 

| | Positive Prediction | Negative Predition |
| :---: | :---: | :---: |
| Positive (P) | True Positive (TP) | False Negative (FN) |
| Negative (N) | False Positive (FP) | True Negative (TN) |

**Precision**: Accuracy of positive predictions, $\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}$

**Recall**: Ability to find all the positive instances, $\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}$

**Accuracy**: $\frac{\mathrm{TP}+\mathrm{TN}}{\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{TN}}$

**F1 sorce**: Harmonic mean of precision and recall, $\frac{2}{\mathrm{Precision}^{-1}+\mathrm{Recall}^{-1}}$


let $y$ be the true answer, and $y'$ be the prediction.

- **Regression**: 
  - Square error: $\mathbf{error}_{\mathrm{sq}}(y,y')=(y-y')^2$
  - Absolute error: $\mathbf{error}_{\mathrm{abs}}(y,y')=|y-y'|$

- **Classification**:
  - Misclassification error: $\mathbf{error}_{\mathrm{mis}}(y,y')=\mathbb{I}\{y\ne y'\}$
  - Weighted misclassification error: If false positive are $\beta$ times worse than false negatives: $\mathbf{error}_{\mathrm{beta}}(y,y')=\beta \mathbb{I}\{y'=1,y=-1\}+(1-\beta)\mathbb{I}\{y'=-1,y=1\}$
  - Balanced error rate: For data with $n_+$ postive samples and $n_-$ negative samples: 

$$
\begin{aligned}
&\text{For one sample: }\\
&\mathbf{error}_{\mathrm{bal}}(y,y')=\frac{\frac{1}{2}(n_++n_-)\cdot \mathbb{I}(y\ne y')}{n_+\mathbb{I}(y)} \\
&\text{For a whole dataset: }\\
&\mathrm{BER}=\frac{1}{2}\left(\frac{\mathrm{FN}}{\mathrm{TP}+\mathrm{FN}}+\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}\right)
\end{aligned}
$$

### Validation
Split dateset into training set $\mathcal{D}_\mathrm{train}$ and validation set $\mathcal{D}_\mathrm{valid}$, and pick different model classes $\phi_1, \phi_2, \dots, \phi_m$; Then trains these $m$ models with $\mathcal{D}_\mathrm{train}$ to get hypothesises $h_1, h_2, \dots, h_m$. Evaluate hypothesis on $\mathcal{D}_{\mathrm{valid}}$ and chooes the lowest:

$$
h^*=\arg \min_{h_i}\left\{\mathbb{E}_{\mathcal{D}_{\mathrm{valid}}}(h)\right\}
$$

**Cross-validation** : For each possible split of data: $(\mathcal{D}_1, \mathcal{D}'_1), (\mathcal{D}_2, \mathcal{D}'_2), \dots, (\mathcal{D}_K, \mathcal{D}'_K)$ , train $i$-th model $\phi_i$ and get hypothesis $h_{\phi_i, \mathcal{D}_j}$ , then estimate with 

$$
\phi^*=\arg\min_{\phi_i}\frac{1}{K}\sum_{i=1}^K E_{\mathcal{D}'_j}\left(h_{\phi_i, \mathcal{D}_j}\right)
$$

- Leave-one-out cross-validation: $|\mathcal{D}'|=1$, accurate, but slow, e.g. validation of SVM.
- $n$-fold cross-validation: $|\mathcal{D}'|=\frac{1}{n}(\text{size of entire dataset})$, decently accurate, not too slow.

### Objective Function

$$
\text{Objective Function}=\text{Error} + \text{Regularizer}
$$

**Pros & Cons** of Regularization:

- **Pros**:
  - reduce variance of the model (make the model more stable) 使模型的表现更稳定
  - Prevent overfitting 防止过拟合
  - Impose prior structural knowledge 施加先验的结构知识
  - Improve interpretability: simpler model, automatic feature selection 增强可解释性
- **Cons**: 
  - *Gauss-Markov* theorem: Least squares is the best linear unbiased estimator
  - regularization increases bias

$$
\text{expected prediction error}=\text{bias}^2+\text{noise}+\color{red}\text{variance}
$$

- $\text{bias}^2$: average prediction error over all datasets (squared)
- $\text{variance}$: how much solutions for different datasets vary (stability)
- $\text{noise}$: deviation of measurements from the treu value (unavoidable error)

**No free lunch! Theorem**: low bias comes with high variance and low variances comes with high bias! $\text{bias}$ 和 $\text{variance}$ 负相关

- Too high bias $\Rightarrow$ cannot fit the data well, due to the restricted solutions
- Too low bias $\Rightarrow$ have too high variance, sensitive to data changes (overfitting)

So we need to find the sweet spot between $\text{bias}$ and $\text{variance}$