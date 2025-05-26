# Dimensionality Reduction

## Principal Component Analysis (PCA)

Finds new axes (principal components) that capture most important patterns in the data 
- Maximize variance 最大化方差，如果新空间中折叠或者压缩 (collapsed) 了，那么就会失去信息
- Minimize reconstruction error 最小化重构误差

$\mathrm{x}_i\approx \sum_{j=1}^K z_{i,j} \mathbf{v}_j$ where $\mathbf{v}_j$ are the best directions to capture importance patterns

$$
\begin{aligned}
&\textbf{Input}~\mathbf{X}\in\mathbb{R}^{n\times d} (n \text{ samples}, d \text{ features}), \text{target dimension } k \\
1. &\text{Standardize the Data 将数据标准化 } \mathrm{X}' \leftarrow \left(\mathrm{V}^{\frac{1}{2}}\right)^{-1}(\mathrm{X}-\boldsymbol{\mu}) \\
2. &\text{Compute Convariance Matrix 计算协方差 } \Sigma = \frac{1}{n}\mathrm{X}'^\top \mathrm{X}' \\
3. &\text{Eigenvalue Decomposition of } \Sigma \text{ 计算特征值和单位特征向量 } \\
4. &\text{Select Principal Components} \text{ 将前 } k \text{ 大的特征值对应的单位特征向量取出, 组成矩阵 } \mathrm{W}\in\mathbb{R}^{d\times k} \\
5. &\text{Project Data} \text{ 使用 } \mathrm{Z}=\mathrm{X}'\mathrm{W} \text{ 映射向量}
\end{aligned}
$$

- **Pro**:
  - Linear method, simple 先行操作，简单
  - Computationally efficient for large datasets 对大数据集的计算效率高
  - Reduces multi-collinearity effectitively 有效降低多重共线性
- **Cons**:
  - Assumes linear relationships in data 假设数据有线性关系
  - Less effective for preserving local structure 不能有效保留局部结构
  - May loss critical information if variance is not representative item 如果方差不具有代表性，那么就会损失很多信息
  - Sensitive to outliers 对异常点敏感

### Variance Maximization 最大化方差

对于每一维都有方差最大化

$$
\begin{aligned}
&\mathrm{Var}[\mathrm{Z}]=\mathrm{Var}[\mathrm{X'}\mathrm{W}]=\mathrm{W}^\top\mathrm{Var}[\mathrm{X}']\mathrm{W}=\mathrm{W}^\top\Sigma\mathrm{W} \\
\text{For one dimension } &\mathrm{Var}[\mathrm{X}'\mathbf{u}]=\mathbf{u}^\top \mathrm{Var}[\mathrm{X}']\mathbf{u}=\mathbf{u}^\top \Sigma\mathbf{u}
\end{aligned}
$$

Then the optimization problem

$$
\max_{\mathbf{u}} \mathbf{u}^\top \Sigma \mathbf{u} \quad \text{subject to } \mathbf{u}^\top \mathbf{u}=1
$$

By *Rayleigh quotient* , we have that top-$k$ $\mathbf{u}$ s are unit eigenvectors corresponds to top-$k$ eigenvalues 根据某定理可以得知， 最终一定会选择最大的特征值对应的单位特征向量

### Reconstruction Error Minimization 最小化重建误差

projected points: $\mathrm{Z}=\mathrm{X}\mathrm{W}=[\hat{\mathbf{x}}_i]$; reconstructed points: $\hat{\mathrm{X}}=\mathrm{Z}\mathrm{W}^\top=\mathrm{X}\mathrm{W}\mathrm{W}^\top$

By *Pythagorean theorem* , the reconstruction error is 

$$
\underbrace{\sum_{i=1}^n \left\Vert\mathbf{x}_i-\hat{\mathbf{x}}_i\right\Vert^2}_{\text{Reconstruction Error}}=\underbrace{\sum_{i=1}^n \left\Vert\mathbf{x}_i\right\Vert^2}_{\text{constant}} + \underbrace{\sum_{i=1}^n \left\Vert\hat{\mathrm{x}}_i\right\Vert^2}_{\text{Variance}}
$$

最小化重构误差 $\Leftrightarrow$ 最大化投影方差

### 标准化

如果不减去平均值，那么计算协方差的时候要减去 $\boldsymbol{\mu}$ 

如果不除以方差，那么较高方差的一维就会主导 principal componenet, 那么最终结果就对每一维的单位很敏感。

### Multidimensional Scaling (MDS)

尽可能保留邻接矩阵 Proximity Matrix 的属性

#### Classical MDS

Bases on Euclidean distances

