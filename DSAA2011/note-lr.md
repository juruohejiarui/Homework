# Linear Regression

## $d$-Dimension Vector to $\mathbb{R}$

$m$: size of dataset 数据点个数

$d$: dimension/length of each feature vector (input) 输入的维度/长度

$y_i$: scaler or real-valued target/output 输出

Design a function $f_{\bf{W}, b}(\bf{x})$ i.e.

$$
f_{\bf{w}, b}(\bf{x})=\bf{w}^{\top}\bf{x}+b
$$

- weight: $\bf{w}\in \mathbb{R}^d$
- bias/offset: $b\in \mathbb{R}$

written in vector form:

$$
f_{\bf{w}, b}=\begin{bmatrix}
b\\
\bf{w}
\end{bmatrix}^{\top}\begin{bmatrix}
1\\
\bf{x}
\end{bmatrix}
$$

let $\overline {\bf{x}} = \begin{bmatrix}1\\\bf{x}\end{bmatrix}, \overline {\bf{w}} = \begin{bmatrix}b\\\bf{w}\end{bmatrix}$, then we have $f_{\bf{w}, b}(\bf{x})={\overline{\bf{w}}}^{\top}\overline{\bf{x}}=\overline{\bf{x}}^{\top}\overline{\bf{w}}$

### Objective (loss) Function 
let $\mathrm{X}=[\overline{\bf{x}}^{\top}_1, \overline{\bf{x}}^{\top}_2, \dots, \overline{\bf{x}}^{\top}_m]^{\top} \in \mathbb{R}^{m\times (d+1)}$ and $\bf{y}=[y_1, y_2, \dots, y_m]^{\top}\in \mathbb{R}^m$ , then loss function is 

$$
\begin{aligned}
\mathrm{Loss}(\bf{w}, b)
&= \frac{1}{m}\sum_{i=1}^m (f_{\bf{w}, b}(\bf{x}_i)-y_i)^2 \\
&= \frac{1}{m}\sum_{i=1}^m ({\overline{\bf{w}}}^{\top}\overline{\bf{x}}_i-y_i)^2 \\
&= \frac{1}{m}(\mathrm{X}\overline{\bf{w}}-\bf{y})^{\top}(\mathrm{X}\overline{\bf{w}}-\bf{y})
\end{aligned}
$$

then let $J(\overline{\bf{w}})=(\mathrm{X}\overline{\bf{w}}-\bf{y})^{\top}(\mathrm{X}\overline{\bf{w}}-\bf{y})=2\overline{\bf{w}}^{\top}\mathrm{X}^{\top}\mathrm{X}\overline{\bf{w}}-2\overline{\bf{w}}^{\top}\mathrm{X}^{\top}\bf{y}+\bf{y}^{\top}\bf{y}$.
the derivative is $\frac{\mathrm{d}J}{\mathrm{d}\overline{\bf{w}}}=2\overline{\bf{w}}^{\top}\mathrm{X}^{\top}\mathrm{X}-2\bf{y}^{\top}\mathrm{X}$ . Then we have $\color{red}{\overline{\bf{w}}^*}=(\mathrm{X}^{\top}\mathrm{X})^{-1}\mathrm{X}^{\top}\bf{y}$

### MLE
let the likelihood function be

$$
L(\overline{\bf{w}}, \sigma^2 | \{y_i, \bf{x}_i\})=\frac{1}{\sqrt{2\pi \sigma^2}}\prod_{i=1}^m \exp\left(-\frac{(y_i - \overline{\bf{w}}^\top \bf{x}_i)^2}{2\sigma^2}\right)
$$

MLE of distribution of error: $e_i\sim \mathcal{N}(0, \hat{\sigma}^2)$

## $d$-Dimensional Vector to $h$-Dimensional Vector

$h$: dimension/length of output vector

$\bf{y}$: output $\in \mathbb{R}^h$

Then $\mathrm{W}\in \mathbb{R}^{h\times d}, \bf{b}\in \mathbb{R}^d, \overline{\mathrm{W}}=\begin{bmatrix}\bf{b}^{\top}\\\mathrm{W}\end{bmatrix}\in \mathbb{R}^{(d+1)\times h}$ 

then $f_{\mathrm{W}, \bf{d}}(\bf{x})=\mathrm{W}^\top\bf{x}+\bf{b}$

then we have

$$
\hat{\mathrm{w}}=(\overline{\bf{X}}^\top\overline{\bf{X}})^{-1}\overline{\bf{X}}^\top\bf{y}\\
\hat{\sigma}^2=\frac{1}{m}\sum_{i=1}^m \left(y_i-\begin{bmatrix}1&\bf{x}_i^\top\end{bmatrix}\hat{\mathrm{w}}\right)^2=\frac{1}{m}(\overline{\mathrm{X}}\hat{\bf{w}}-\bf{y})^\top(\overline{\mathrm{X}}\hat{\bf{w}}-\bf{y})
$$

### Objective (Loss) Function

let $\overline{\mathrm{Y}}=[\bf{y}_1^\top, \bf{y}_2^\top, \dots, \bf{y}_m^\top]^\top \in \mathbb{R}^{m\times h}$ and $\mathrm{X}$ is the same as the previous definition.

then

$$
\mathrm{Loss}(\overline{\mathrm{W}})=\sum_{k=1}^h (\mathrm{X}\overline{\mathrm{W}}^{(k)}-\mathrm{Y}^{(k)})^\top(\mathrm{X}\overline{\mathrm{W}}^{(k)}-\mathrm{Y}^{(k)})
$$

Then the least squares solution is $\overline{\mathrm{W}}^*=\arg\min_{\mathrm{W},\bf{b}} \mathrm{Loss}(\overline{\mathrm{W}})=\color{red}{(\mathrm{X}^\top\mathrm{X})^{-1}\mathrm{X}^\top\mathrm{Y}}$

we need to guarantee that $\mathrm{X}$ is full rank to make $(\mathrm{X}^\top\mathrm{X})^{-1}$ exist. 

## For Binary Classification

let $\mathrm{sign}(x)=\begin{cases}1 & \text{if } x>0 \\ 0 &\text{if }x=0 \\ -1 &\text{otherwise}\end{cases}$

we just need to modify the output with 

$$
g_{\bf{w}, b}(\bf{x})=\mathrm{sign}(f_{\bf{w},b}(\bf{x}))=\mathrm{sign}\left(\overline{\bf{x}}^\top\bf{w}\right)
$$

PS: output $=0$ declares error.

## For Multi-class Classification

Apply **one-hot encoding** . Assume that there are $h$ labels, then for one output $\bf{y}\in\mathbb{R}^h$, which is $t$-th label, then we have $\forall i\in \mathbb{Z}\cap [1, h]\backslash\{x\}, y_i=0, y_t=1$ . 用人话来说就是在正确的标签上面打一个 $1$, 其余位置为 $0$. 

Then the output label can be 

$$
\mathrm{output}=\arg\max_{k\in \{1, 2, ... h\}} \left\{\overline{\bf{x}}^\top \overline{\mathrm{W}}^{(k)}\right\}
$$

选取值最大的一个

## Polynomial Regression

Treat different polynomial items of one input as items of different dimensions of input.

Assume that we have an $p$-ordered input, then $d'=\binom{d+p}{p}$. And the input can be like this:

$$
\bf{p}=\begin{bmatrix}1&x_1&x_2&\dots&x_1x_2&\dots&x_1x_2x_3&\dots&\prod_{i=1}^p x_i&\dots\end{bmatrix}^\top \in \mathbb{R}^{\binom{d+p}{p}}
$$

Then for 1-d output, weight vector is $\bf{w}\in\mathbb{R}^{\binom{d+p}{p}}$ 
For multi-dimensional output, weight matrix is $\mathrm{W}\in \mathbb{R}^{\binom{d+p}{p}\times h}$

and

$$
y=\bf{p}^\top \bf{w} \\
\bf{y}=\bf{p}^\top \mathrm{W}
$$

Then we have $\mathrm{P}=\begin{bmatrix}\bf{p}_1^\top&\bf{p}_2^\top&\dots&\bf{p}_n^\top\end{bmatrix}^\top\in \mathbb{R}^{m\times \binom{d+p}{p}}$



Similarly, we have 

$$
\begin{aligned}
\bf{w}^*&=\mathrm{P}^\top(\mathrm{P}\mathrm{P}^\top)^{-1}\bf{y} \\
\mathrm{W}^*&=\mathrm{P}^\top(\mathrm{P}\mathrm{P}^\top)^{-1}\mathrm{Y}
\end{aligned}
$$

## Ridge Regresion

Assume that when the data have many variables/attributes and the dimension ($d$) is large, and few samples ($m$) is small. Then $\mathrm{X}\in \mathbb{R}^{m\times (d+1)}$ which is hard to be full rank, then $\mathrm{X}^\top\mathrm{X}$ may not exist.

Then modify the loss function with a positive constant $\lambda$ : 

$$
\begin{aligned}
J(\overline{\bf{w}})&=\sum_{i=1}^m (\overline{\bf{x}}^\top_i\overline{\bf{w}}-y_i)^2+\lambda\sum_{i=1}^{d+1}\overline{w}_i^2 \\
&=(\mathrm{X}\overline{\bf{w}}-\bf{y})^\top(\mathrm{X}\overline{\bf{w}}-\bf{y})+\lambda\overline{\bf{w}}^\top\overline{\bf{w}}
\end{aligned}
$$

Then we get 

$$
\color{red}{
\overline{\bf{w}}^*=(\mathrm{X}^\top\mathrm{X}+\lambda \mathrm{I}_{d+1})^{-1}\mathrm{X}^\top\bf{y}}
$$

This form is called **Primal Form**. 

where $\mathrm{X}^\top\mathrm{X}+\lambda \mathrm{I}_{d+1}$ is always invertible. This can be proved by applying theorems mentioned in *Mathematical Tools* .

Since $\mathrm{X}^\top\mathrm{X}+\lambda \mathrm{I}_{d+1}\in \mathbb{R}^{(d+1)\times (d+1)}$ can be large and computation of its inverse is costly, we need **Dual Form**, which is :


$$
\color{red}{
\overline{\bf{w}}^*=\mathrm{X}^\top(\mathrm{X}\mathrm{X}^\top+\lambda \mathrm{I}_m)^{-1}\bf{y}}
$$

Which is equivalent to **Primal Form**. It can be obtained from **Primal Form** applying *Woodbury Formula*.

PS: polynomial ridge regression is which just replace $\mathrm{X}$ by $\mathrm{P}$ .

## Logistic Regression

Give an input vector $\bf{x}\in \mathbb{R}^{d}$ . The parameters of logistic regression are $\vec{\theta}\in \mathbb{R}^d, \theta_0\in \mathbb{R}$ . let $g$ be the sigmoid function.

$$
\mathrm{Pr}\left(y=1|\bf{x},\vec{\theta},\theta_0\right)=g\left(\left<\vec\theta,\bf{x}\right>+\theta_0\right)=\frac{1}{1 + \exp\left(-\left<\vec\theta,\bf{x}\right>-\theta_0\right)}
$$

More generally, we have 

$$
\mathrm{Pr}\left(y=y_0|\bf{x}=\bf{x}_0,\vec\theta,\theta_0\right)=g\left(y_0\left(\left<\vec\theta,\bf{x}_0\right>+\theta_0\right)\right)
$$

where $y_0\in\{-1, 1\}$

### Why?
**logs-odd** should be linear function:

$$
\log\frac{\mathrm{Pr}\left(y=1|\bf{x},\vec\theta,\theta_0\right)}{\mathrm{Pr}\left(y=-1|\bf{x},\vec\theta,\theta_0\right)}=\left<\vec\theta,\bf{x}\right>+\theta_0
$$

Decision boundary:
where $\left<\vec\theta,\bf{x}\right>+\theta_0=0$

### MLE

Likelihood function of $(\mathrm{x_0},y_0)$ of given $(\vec\theta, \theta_0)$: 

$$
L(\vec\theta, \theta_0|\mathrm{x_0},y_0)=\mathrm{Pr}(y=y_0|\bf{x}=\bf{x}_0,\vec\theta,\theta_0)
$$

For likelihood function data set $\mathcal{D}=\{(\mathrm{x}_i, y_i)_i\}$

$$
L(\vec\theta, \theta_0|\mathcal{D})=\prod_{i=1}^m \mathrm{Pr}(y=y_i|\bf{x}=\bf{x}_i,\vec\theta,\theta_0)
$$

we need to maximum the value of $L(\vec\theta, \theta_0|\mathcal{D})$ .

$$
\begin{aligned}
\arg\max_{\vec\theta, \theta_0}\left\{L(\vec\theta, \theta_0|\mathcal{D})\right\}
&=\arg\max_{\vec\theta, \theta_0}
    \sum_{i=1}^m 
        \log \mathrm{Pr}\left(y=y_i|\bf{x}=\bf{x}_i,\vec\theta,\theta_0\right) \\
&=\arg\min_{\vec\theta, \theta_0}
    \sum_{i=1}^m 
        \log \mathrm{Pr}\left(y=y_i|\bf{x}=\bf{x}_i,\vec\theta,\theta_0\right) \\
&=\arg\min_{\vec\theta, \theta_0}
    \sum_{i=1}^m 
        \log \left[1+\exp\left(-y_i\left(\left<\vec\theta,\bf{x}\right>+\theta_0\right)\right)\right]
\end{aligned}
$$

Denote that 

$$
l(\vec\theta, \theta_0|\mathcal{D})=\sum_{i=1}^m \log \left[1+\exp\left(-y_i\left(\left<\vec\theta,\bf{x}\right>+\theta_0\right)\right)\right]
$$

### Applying Stochastic Gradient Descent

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d}\theta_0}\left(\log \left[1+\exp\left(-y_i\left(\left<\vec\theta,\bf{x}\right>+\theta_0\right)\right)\right]\right)
&= -y_i\frac
    {\exp\left(-y_i\left(\left<\vec\theta,\bf{x}\right>+\theta_0\right)\right)}
    {1+\exp\left(-y_i\left(\left<\vec\theta,\bf{x}\right>+\theta_0\right)\right)} \\
&= \color{red}{-y_i\left[1-\mathrm{Pr}\left(y=y_i|\bf{x}=\bf{x}_i,\vec\theta,\theta_0\right)\right]}\\
\frac{\mathrm{d}}{\mathrm{d}\vec\theta}\left(\log \left[1+\exp\left(-y_i\left(\left<\vec\theta,\bf{x}\right>+\theta_0\right)\right)\right]\right)
&=
\color{red}{-y_i\bf{x}_i\left[1-\mathrm{Pr}\left(y=y_i|\bf{x}=\bf{x}_i,\vec\theta,\theta_0\right)\right]}\\
\end{aligned}
$$

Then 

$$
\frac{\mathrm{d}l(\vec\theta, \theta_0|\mathcal{D})}{\mathrm{d}\theta_0}
= \sum_{i=1}^n 
    -y_i\left[1-\mathrm{Pr}\left(y=y_i|\bf{x}=\bf{x}_i,\vec\theta,\theta_0\right)\right] \\
\frac{\mathrm{d}l(\vec\theta, \theta_0|\mathcal{D})}{\mathrm{d}\vec\theta}
= \sum_{i=1}^n
    -y_i\bf{x}_i\left[1-\mathrm{Pr}\left(y=y_i|\bf{x}=\bf{x}_i,\vec\theta,\theta_0\right)\right]
$$

SGD leads to **no** significant change on average when the gradient of the full objective equals zero.

### Regularization in Logistic Regression 正则化

likelihood function is strictly increasing as function of $y_i\left(\left<\vec\theta, \bf{x}\right>+\theta_0\right)$ . Then we can infinitely scale the parameters to obtains the higher likelihood. This can make the model emphasis too much on the partial features and noise, which leads to overfitting. 参数无限制增大会导致模型过度关注噪声和局部特征，导致过拟合。

e.g. $l_2$-norm :

$$
\arg\min_{\vec\theta, \theta_0}\left\{ \frac{\lambda}{2}\left\Vert\vec\theta\right\Vert^2 + l(\vec\theta, \theta_0|\mathcal{D}) \right\}
$$

### Multi-Class Logistic Regression

If we have $h\ge 2$ classes, replace $\mathrm{Pr}(y=1|\bf{x},\vec\theta,\theta_0)$ by **soft-max** function:

$$
\mathrm{Pr}(y=c|\bf{x},\vec\theta, \theta_0)=\frac
	{\exp\left(\left<\vec\theta_c, \bf{x}\right>+\theta_{0,c}\right)}
	{\sum_{i=1}^h {\exp\left(\left<\vec\theta_i, \bf{x}\right>+\theta_{0,i}\right)}}
$$