# Kernel Functions

A way to handle non-linear data. Data may be separatable in higher dimensions. $\mathbf{x}\mapsto \phi(\mathbf{x})$

In many cases, we do not need to compute $\phi(\mathbf{x})$ explicitly, but only the inner product $K(\mathbf{x}, \mathbf{y})=\langle \phi(\mathbf{x}), \phi(\mathbf{y})\rangle=\phi(\mathbf{x})^\top \phi(\mathbf{y})$.

- **Linear Kernel**: $K(\mathbf{x}, \mathbf{y})=\mathbf{x}^\top\mathbf{y}$
- **Polynomial Kernel**: $K(\mathbf{x}, \mathbf{y})=\left(\mathbf{x}^\top\mathbf{y}+c\right)^d$
- **Radial Basis Function (RBF) Kernel**: $K(\mathbf{x}, \mathbf{y})=\exp\left(-\gamma\left\Vert\mathbf{x}-\mathbf{y}\right\Vert^2\right)$
- **Sigmoid Kernel**: $K(\mathbf{x}, \mathbf{y})=\tanh\left(\alpha\mathbf{x}^\top\mathbf{y}+c\right)$

## SVM 

$$
\begin{aligned}
&\mathcal{L}(\mathbf{w}, b, \lambda)=\sum_{i=1}^n \lambda_i-\frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \lambda_i\lambda_j y_i y_j(\mathbf{x}_i^\top \mathbf{x}_j) \\
\Rightarrow & \mathcal{L}(\mathbf{w}, b, \lambda)=\sum_{i=1}^n \lambda_i - \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \lambda_i\lambda_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \\
\end{aligned}
$$

## Clustering

Centroid :

$$
\phi(\mathbf{c}_i)=\frac{1}{|C_i|}\sum_{\mathbf{x}\in C_i} \phi(\mathbf{x})
$$

Distance between points:

$$
\begin{aligned}
\left\Vert \phi(\mathbf{x}) - \phi(\mathbf{y})\right\Vert^2 &= (\phi(\mathbf{x}) - \phi(\mathbf{y}))^\top (\phi(\mathbf{x})-\phi(\mathbf{y})) \\
&= \phi(\mathbf{x})^\top \phi(\mathbf{x}) - 2\cdot \phi(\mathbf{x})^\top \phi(\mathbf{y})+\phi(\mathbf{y})^\top \phi(\mathbf{y}) \\
&= \color{red} K(\mathbf{x}, \mathbf{x}) - 2K(\mathbf{x}, \mathbf{y}) + K(\mathbf{y}, \mathbf{y}) \\
\end{aligned}
$$

Distance of points to centroid:

$$
\left\Vert \phi(\mathbf{x})-\phi(\mathbf{c}_i)\right\Vert^2 = 1 - \frac{2}{|C_i|}\sum_{\mathbf{y}\in C_i} K(\mathbf{x}, \mathbf{y})+\frac{1}{|C_i|^2}\sum_{\mathbf{y}_1,\mathbf{y}_2\in C_i} K(\mathbf{y}_1, \mathbf{y}_2)
$$