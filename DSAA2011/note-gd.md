# Gradiant Descent

## Standard GD

$$\nabla_\mathbf{w}C(\mathbf{w})=\begin{bmatrix}
\frac{\partial C}{\partial \mathbf{w}_1} & \frac{\partial C}{\partial \mathbf{w}_2} & \dots & \frac{\partial C}{\partial \mathbf{w}_d}
\end{bmatrix}
$$

PS: there is transport ($\top$) in the slide, which does not fit the definition in AIAA2711 ...

### Persudo Code
$$
\begin{aligned}
&\text{Algorithm GD} \\
&~~~~\text{initialize }w_0\text{ and learning rate }\eta>0 \\
&~~~~k\leftarrow 0 \\
&~~~~\text{while } true\text{ do} \\
&~~~~~~~~\mathbf{w}_{k+1}\leftarrow\mathbf{w}_k-\eta\nabla_{\mathbf{w}}C(\mathbf{w}_k) \\
&~~~~~~~~\text{if }coverage\text{ then} \\
&~~~~~~~~~~~~\text{return }\mathrm{w}_{k+1} \\
&~~~~~~~~\text{end if} \\
&~~~~\text{end while}
\end{aligned}
$$

**impact** of $\eta$ :

- $\eta$ is too small: Converge slowly
- $\eta$ is too large: 
  - overshoot the loca
  - converge slowly to the minimum or may never converge

Posible convergence criteria:

- set maximum iteration $k_{\max}$ 
- check the percentage or absolute change in objective function $C$ below a threshold
- check the percentage or absolute change in parameter $\mathbf{w}$ below a threshold

- 设置最大迭代次数
- 当参数或者梯度小于某一个限定值，则停止。

PS: GD can only find local minimum

- gradient $=0$ at local minimum, and $\mathbf{w}$ will not change after reach local minimum.

## Variation of GD

### Change Learning Rate

**Decreasing learning rate**: 

$$
\begin{aligned}
\eta_{k+1}&\leftarrow\eta_k\cdot\alpha \text{ or } \eta_k -\alpha \\
\mathbf{w}_{k+1}&\leftarrow\mathbf{w}_k-\eta_k\nabla_\mathbf{w}C(\mathbf{w}_k)
\end{aligned}
$$

**Adagrad** (Adaptive Gradient Algorithm) :

$$
\begin{aligned}
& \mathbf{w}_{k+1}\leftarrow \mathbf{w}_k-\eta\cdot (\mathrm{M}_k+\epsilon \mathrm{I})^{-1}\nabla_{\mathbf{w}}(\mathbf{w}_k) \\
\text{where } &\mathrm{M}_k=\mathrm{diag}\left(\sum_{i=1}^k \nabla_\mathbf{w}(\mathbf{w}_i)\odot \nabla_\mathbf{w}(\mathbf{w}_i)\right)
\end{aligned}
$$

- $\mathrm{diag}(\mathbf{v}) \in \mathbb{R}^{n\times n} \text{ where } \mathbf{v}\in\mathbb{R}^n$ : 对角线元素 $\mathrm{diag}(\mathbf{v})_{i,i}$ 是 $v_i$ 
- $\mathbf{a} \odot \mathbf{b} \in \mathbb{R}^n \text{ where } \mathbf{a}, \mathbf{b}\in \mathbb{R}^n$ : 对位乘起来。


### Different Gradient

**Momentum-based GD**:

$$
\begin{aligned}
\mathbf{v}_k&\leftarrow\beta\mathbf{v}_{k-1}+(1-\beta)\nabla_\mathbf{w}C(\mathbf{w}_k) \\
\mathbf{w}_{k+1}&\leftarrow \mathbf{w}_k-\eta\mathbf{v}_k
\end{aligned}
$$

where $\beta$ is the momentum factor, and $\mathbf{v}_0=\nabla_\mathbf{w}(\mathbf{w}_0)$. 

This algorithm can maintaining a moving average of past gradient. It converges fast but can overshoot the minimum.

**Nesterov Acceleared Gradient (NAG)** :

$$
\begin{aligned}
\mathbf{v}_k&\leftarrow\beta \mathbf{v}_{k-1}+\eta\nabla_{\mathbf{w}}(\mathbf{w}_k+\beta\mathbf{v}_{k-1}) \\
\mathbf{w}_{k+1}&\leftarrow\mathbf{w}_k-\mathbf{v}_k
\end{aligned}
$$

Concept: anticipate the direction the optimizer will go in the next step.
- Pros: faster convergence
- Cons: slightly more complex than momentum alone

### View of Data Sheet

**Batch Gradient Descent(BGD)** : Loss function is built using the entire dataset at each iteration 使用整个数据集来计算损失函数的值。 e.g. 所有的loss取平均数

- Pros: converges smoothly to the minimum 收敛较为顺滑
- Cons: computationally expensive, memory intensive, not ideal for online learning (hard to update the model incrementally as new data comes) 计算强度大，内存消耗大，难以支持动态添加新数据

**Stochastic Gradient Descent(SGD)** : use one **randomly chosen sample** to build the loss function at each iteration 每次迭代使用一个随机的样本计算损失函数

- Pros: fast updates, suitable for online learning 快速更新值，对动态添加数据友好
- Cons: noisy update, longer convergence time 更新过程会出现噪声，不稳定，需要的收敛时间更长

**Mini-Batch Gradient Descent(MBGD)**: use some **randomly chosen samples** to build the loss function at each iteration 每次迭代选取一个小样本集合。

- Pros: between BGD and SGD
- Cons: still has noise, hperparamters tuning (choice of batch size)

$$
\begin{aligned}
&\text{Algorithm MBGD} \\
&~~~~\text{initialize }w_0\text{ and learning rate }\eta>0 \\
&~~~~k\leftarrow 0 \\
&~~~~\text{divide }\mathcal{D}\text{ into mini-batches} \\
&~~~~\text{while } true\text{ do} \\
&~~~~~~~~\text{randomly choose one mini-batch } n \\
&~~~~~~~~\text{calculate }\nabla_\mathbf{w}C_n(\mathbf{w}_k) \\
&~~~~~~~~\mathbf{w}_{k+1}\leftarrow\mathbf{w}_k-\eta\nabla_{\mathbf{w}}C_n(\mathbf{w}_k) \\
&~~~~~~~~\text{if }coverage\text{ then} \\
&~~~~~~~~~~~~\text{return }\mathrm{w}_{k+1} \\
&~~~~~~~~\text{end if} \\
&~~~~\text{end while}
\end{aligned}
$$