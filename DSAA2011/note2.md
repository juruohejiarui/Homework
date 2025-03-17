# Linear Regression

## $d$-Dimension Vector to $\mathbb{R}$

$m$: size of dataset 数据点个数
$d$: dimension/length of each feature vector (input) 输入的维度/长度
$y_i$: scaler or real-valued target/output 输出

Design a function $f_{\mathbf{W}, b}(\mathbf{x})$ i.e.
$$
f_{\mathbf{w}, b}(\mathbf{x})=\mathbf{w}^{\top}\mathbf{x}+b
$$
- weight: $\mathbf{w}\in \mathbb{R}^d$
- bias/offset: $b\in \mathbb{R}$

written in vector form:
$$
f_{\mathbf{w}, b}=\begin{bmatrix}
b\\
\mathbf{w}
\end{bmatrix}^{\top}\begin{bmatrix}
1\\
\mathbf{x}
\end{bmatrix}
$$

let $\overline {\mathbf{x}} = \begin{bmatrix}1\\\mathbf{x}\end{bmatrix}, \overline {\mathbf{w}} = \begin{bmatrix}b\\\mathbf{w}\end{bmatrix}$, then we have $f_{\mathbf{w}, b}(\mathbf{x})={\overline{\mathbf{w}}}^{\top}\overline{\mathbf{x}}=\overline{\mathbf{x}}^{\top}\overline{\mathbf{w}}$

### Objective (loss) Function 
let $\mathrm{X}=[\overline{\mathbf{x}}^{\top}_1, \overline{\mathbf{x}}^{\top}_2, \dots, \overline{\mathbf{x}}^{\top}_m]^{\top} \in \mathbb{R}^{m\times (d+1)}$ and $\mathbf{y}=[y_1, y_2, \dots, y_m]^{\top}\in \mathbb{R}^m$ , then loss function is 
$$
\begin{aligned}
\mathrm{Loss}(\mathbf{w}, b)
&= \frac{1}{m}\sum_{i=1}^m (f_{\mathbf{w}, b}(\mathbf{x}_i)-y_i)^2 \\
&= \frac{1}{m}\sum_{i=1}^m ({\overline{\mathbf{w}}}^{\top}\overline{\mathbf{x}}_i-y_i)^2 \\
&= \frac{1}{m}(\mathrm{X}\overline{\mathbf{w}}-\mathbf{y})^{\top}(\mathrm{X}\overline{\mathbf{w}}-\mathbf{y})
\end{aligned}
$$

then let $J(\overline{\mathbf{w}})=(\mathrm{X}\overline{\mathbf{w}}-\mathbf{y})^{\top}(\mathrm{X}\overline{\mathbf{w}}-\mathbf{y})=2\overline{\mathbf{w}}^{\top}\mathrm{X}^{\top}\mathrm{X}\overline{\mathbf{w}}-2\overline{\mathbf{w}}^{\top}\mathrm{X}^{\top}\mathbf{y}+\mathbf{y}^{\top}\mathbf{y}$.
the derivative is $\frac{\mathrm{d}J}{\mathrm{d}\overline{\mathbf{w}}}=2\overline{\mathbf{w}}^{\top}\mathrm{X}^{\top}\mathrm{X}-2\mathbf{y}^{\top}\mathrm{X}$
then we have $\color{red}{\overline{\mathbf{w}}^*}=(\mathrm{X}^{\top}\mathrm{X})^{-1}\mathrm{X}^{\top}\mathbf{y}$

### MLE
let
$$
L(\overline{\mathbf{w}}, \sigma^2 | {y_i, \mathbf{x}_i})=\frac{1}{\sqrt{2\pi \sigma^2}}\prod_{i=1}^m \exp\left(-\frac{(y_i - \overline{\mathbf{w}}^\top \mathbf{x}_i)^2}{2\sigma^2}\right)
$$

MLE of distribution of error: $e_i\sim \mathcal{N}(0, \hat{\sigma}^2)$

## $d$-Dimensional Vector to $h$-Dimensional Vector

$h$: dimension/length of output vector
$\mathbf{y}$: output $\in \mathbb{R}^h$

Then $\mathrm{W}\in \mathbb{R}^{h\times d}, \mathbf{b}\in \mathbb{R}^d, \overline{\mathrm{W}}=\begin{bmatrix}\mathbf{b}^{\top}\\\mathrm{W}\end{bmatrix}\in \mathbb{R}^{(d+1)\times h}$ 

then $f_{\mathrm{W}, \mathbf{d}}(\mathbf{x})=\mathrm{W}\mathbf{x}+\mathbf{b}$

then we have
$$
\hat{\mathrm{w}}=(\overline{\mathbf{X}}^\top\overline{\mathbf{X}})^{-1}\overline{\mathbf{X}}^\top\mathbf{y}\\
\hat{\sigma}^2=\frac{1}{m}\sum_{i=1}^m \left(y_i-\begin{bmatrix}1&\mathbf{x}_i^\top\end{bmatrix}\hat{\mathrm{w}}\right)^2=\frac{1}{m}(\overline{\mathrm{X}}\hat{\mathbf{w}}-\mathbf{y})^\top(\overline{\mathrm{X}}\hat{\mathbf{w}}-\mathbf{y})
$$

### Objective (Loss) Function

let $\overline{\mathrm{Y}}=[\mathbf{y}_1^\top, \mathbf{y}_2^\top, \dots, \mathbf{y}_m^\top]^\top \in \mathbb{R}^{m\times h}$ and $\mathrm{X}$ is the same as the previous definition.

then
$$
\mathrm{Loss}(\overline{\mathrm{W}})=\sum_{k=1}^h (\mathrm{X}\overline{\mathrm{W}}^{(k)}-\mathrm{Y}^{(k)})^\top(\mathrm{X}\overline{\mathrm{W}}^{(k)}-\mathrm{Y}^{(k)})
$$

Then the least squares solution is $\overline{\mathrm{W}}^*=\argmin_{\mathrm{W},\mathbf{b}} \mathrm{Loss}(\overline{\mathrm{W}})=\color{red}{(\mathrm{X}^\top\mathrm{X})^{-1}\mathrm{X}^\top\mathrm{Y}}$

we need to guarantee that $\mathrm{X}$ is full rank to make $(\mathrm{X}^\top\mathrm{X})^{-1}$ exist. 