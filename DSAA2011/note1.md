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
\begin{aligned}
\hat{\theta}_{\mathrm{ML}} &= \argmax_{\theta} p_{X}(x_1, x_2, \dots, x_n; \theta) \\
&=\argmax_{\theta} \log p_{X}(x_1, x_2, \dots, x_n; \theta)
\end{aligned}
$$

if $x_1, x_2, \dots, x_n$ are independent, then:
$$
\begin{aligned}
\hat{\theta}_{\mathrm{ML}} &= \argmax_{\theta} \prod_{i=1}^n p_{X}(x_i; \theta) \\
&= \argmax_{\theta} \sum_{i=1}^n \log p_{X}(x_i; \theta)
\end{aligned}
$$

MLE can be biased （最大似然估计不一定是无偏的）

### Consistency 
$\hat\theta_{\mathrm{ML}}\xrightarrow{p} \theta$ as $n\rightarrow \infty$ . We say $X_n \xrightarrow{p} X$ as $n\rightarrow \infty$ if 
$$
\lim_{n\rightarrow}\mathrm{Pr}(\|X_n-X\|>\epsilon) = 0, \forall \epsilon\in \mathbb{R}
$$

### Asymptotic Normality 渐进正态
we have $\sqrt{n} \left(\hat\theta_{\mathrm{ML}} - \theta\right)\xrightarrow{d} \mathcal{N}(0, I(\theta)^{-1})$ as $n\rightarrow \infty$, where $I(\theta)$ is fisher information of $\theta$ (larger information $\Longrightarrow$ smaller variance)

We say $X_n\xrightarrow{d}X$ as $n\rightarrow \infty$ if 
$$
\lim_{n\rightarrow \infty}F_{X_n}(x)=F(x)~\forall x \text{  where }f\text{ is continuous}
$$