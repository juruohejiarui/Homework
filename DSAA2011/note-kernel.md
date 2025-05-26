# Kernel Functions

A way to handle non-linear data. Data may be separatable in higher dimensions. $\bf{x}\mapsto \phi(\bf{x})$

In many cases, we do not need to compute $\phi(\bf{x})$ explicitly, but only the inner product $K(\bf{x}, \bf{y})=\langle \phi(\bf{x}), \phi(\bf{y})\rangle=\phi(\bf{x})^\top \phi(\bf{y})$.

- **Linear Kernel**: $K(\bf{x}, \bf{y})=\bf{x}^\top\bf{y}$
- **Polynomial Kernel**: $K(\bf{x}, \bf{y})=\left(\bf{x}^\top\bf{y}+c\right)^d$
- **Radial Basis Function (RBF) Kernel**: $K(\bf{x}, \bf{y})=\exp\left(-\gamma\left\Vert\bf{x}-\bf{y}\right\Vert^2\right)$
- **Sigmoid Kernel**: $K(\bf{x}, \bf{y})=\tanh\left(\alpha\bf{x}^\top\bf{y}+c\right)$

## SVM 

$$
\begin{aligned}
&\mathcal{L}(\bf{w}, b, \lambda)=\sum_{i=1}^n \lambda_i-\frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \lambda_i\lambda_j y_i y_j(\bf{x}_i^\top \bf{x}_j) \\
\Rightarrow & \mathcal{L}(\bf{w}, b, \lambda)=\sum_{i=1}^n \lambda_i - \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \lambda_i\lambda_j y_i y_j K(\bf{x}_i, \bf{x}_j) \\
\end{aligned}
$$

## Clustering

Centroid :

$$
\phi(\bf{c}_i)=\frac{1}{|C_i|}\sum_{\bf{x}\in C_i} \phi(\bf{x})
$$

Distance between points:

$$
\begin{aligned}
\left\Vert \phi(\bf{x}) - \phi(\bf{y})\right\Vert^2 &= (\phi(\bf{x}) - \phi(\bf{y}))^\top (\phi(\bf{x})-\phi(\bf{y})) \\
&= \phi(\bf{x})^\top \phi(\bf{x}) - 2\cdot \phi(\bf{x})^\top \phi(\bf{y})+\phi(\bf{y})^\top \phi(\bf{y}) \\
&= \color{red} K(\bf{x}, \bf{x}) - 2K(\bf{x}, \bf{y}) + K(\bf{y}, \bf{y}) \\
\end{aligned}
$$

Distance of points to centroid:

$$
\left\Vert \phi(\bf{x})-\phi(\bf{c}_i)\right\Vert^2 = 1 - \frac{2}{|C_i|}\sum_{\bf{y}\in C_i} K(\bf{x}, \bf{y})+\frac{1}{|C_i|^2}\sum_{\bf{y}_1,\bf{y}_2\in C_i} K(\bf{y}_1, \bf{y}_2)
$$