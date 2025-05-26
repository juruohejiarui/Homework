# Dimensionality Reduction

## Principal Component Analysis (PCA)

Finds new axes (principal components) that capture most important patterns in the data 
- Maximize variance 最大化方差，如果新空间中折叠或者压缩 (collapsed) 了，那么就会失去信息
- Minimize reconstruction error 最小化重构错误

$\mathrm{x}_i\approx \sum_{j=1}^K z_{i,j} \bf{v}_j$ where $\bf{v}_j$ are the best directions to capture importance patterns

$$
\begin{aligned}
&\textbf{Input}~\bf{X}\in\mathbb{R}^{n\times d} (n \text{ samples}, d \text{ features}), \text{target dimension } k \\
1. &\text{Standardize the Data 将数据标准化 } \mathrm{X}' \leftarrow \left(\mathrm{V}^{\frac{1}{2}}\right)^{-1}(\mathrm{X}-\boldsymbol{\mu}) \\
2. &\text{Compute Convariance Matrix 计算协方差 } \Sigma = \frac{1}{n}\mathrm{X}'^\top \mathrm{X}' \\
3. &\text{Eigenvalue Decomposition of } \Sigma \text{ 计算特征值和单位特征向量 } \\
4. &\text{Select Principal Components} \text{ 将前 } k \text{ 大的特征值对应的单位特征向量取出, 组成矩阵 } \mathrm{W}\in\mathbb{R}^{d\times k} \\
5. &\text{Project Data} \text{ 使用 } \mathrm{Z}=\mathrm{X}'\mathrm{W} \text{ 映射向量}
\end{aligned}
$$

### Variance Maximization 最大化方差

对于每一维都有方差最大化

$$
\begin{aligned}
&\mathrm{Var}[\mathrm{Z}]=\mathrm{Var}[\mathrm{X'}\mathrm{W}]=\mathrm{W}^\top\mathrm{Var}[\mathrm{X}']\mathrm{W}=\mathrm{W}^\top\Sigma\mathrm{W} \\
\text{For one dimension } &\mathrm{Var}[\mathrm{X}'\bf{u}]=\bf{u}^\top \mathrm{Var}[\mathrm{X}']\bf{u}=\bf{u}^\top \Sigma\bf{u}
\end{aligned}
$$

Then the optimization problem

$$
\max_{\bf{u}} \bf{u}^\top \Sigma \bf{u} \quad \text{subject to } \bf{u}^\top \bf{u}=1
$$