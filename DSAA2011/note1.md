# Mathematical Tools
## Bayes's rule 贝叶斯
$\mathcal{X}, \mathcal{Y}$ : event space
$$
p(y|x)=\frac{p(x|y)p(y)}{\sum_{y'\in \mathcal{Y}}p(x|y')p(y')}
$$

## Expectation 期望
- Discrete case 离散情况：
$$
\mathbb{E} Y= \mathbb{E}[g(X)]=\sum_{y\in \mathcal{Y}} y\cdot p_Y(y)=\sum_{x\in \mathcal{X}} g(x)p_X(x)
$$
- Continuous case 连续情况：
where $f$ is the probability density funtion 概率密度函数
$$
\mathbb{E} Y=\mathbb{E}[g(X)]=\int_{\mathbb{R}} yf_Y(y)\mathrm{d}y=\int_{\mathbb{R}}g(x)f_X(x)\mathrm{d}x
$$

## Maximum Likelihood Estimation (MLE) 最大似然估计
Assume we get a probability function $p_X(x; \theta)$ with unconfirmed parameters, $\theta$ . Additionally, there is a data set $x=\{x_1,x_2,\dots,x_n\}$ , then the MLE of $\theta$ is :
$$
\hat{\theta}_{\mathrm{ML}} = \argmax_{\theta} p_{X}(x_1, x_2, \dots, x_n; \theta)
$$