# Support Vector Machine (SVM) 支持向量机

## Preparation

Given a set of data $\mathcal{D}=\{(\mathbf{x}_i,y_i)\in \mathbb{R}^d\times \{-1,1\}:i=1,2,...n\}$

For any $\mathbf{x},y$ , correct classification by classifier $f_{\vec\theta}(\mathbf{x})=\mathrm{sign}\left({\vec\theta}^\top\mathbf{x}\right)$ means that :

$$
\begin{cases}
{\vec\theta}^\top\mathbf{x}>0,y=+1 \\
{\vec\theta}^\top\mathbf{x}<0,y=-1 \\
\end{cases}
\Leftrightarrow y=\mathrm{sign}\left({\vec\theta}^\top\mathbf{x}\right)
\Leftrightarrow y{\vec\theta}^\top\mathbf{x}>0
$$

### Definitions: 
**Margin of Classifier**: The margin of classifier $f_{\vec\theta}$ on sample $(\mathbf{x},y)$ is $y{\vec\theta}\mathbf{x}$ or $y\left<{\vec\theta}, \mathbf{x}\right>$
- **Positive Margin**: $(\mathbf{x},y)$ is correctly classified by $\vec{\theta}$ . 
- **Negative Margin**: $(\mathbf{x},y)$ is not correctly classified by $\vec{\theta}$
- **Bigger Margin**: $(\mathbf{x},y)$ is more correctly clasified by $\vec{\theta}$

**Linearly separable**: The dataset $\mathcal{D}=\{(\mathbf{x}_i,y_i)\in \mathbb{R}^d\times \{-1,1\}:i=1,2,...n\}$ is linearly separable if there exists some $\vec{\theta}$ such that 

$$
y_i=\mathrm{sign}\left({\vec{\theta}}^\top \mathbf{x}\right)\text{ or }y_i{\vec{\theta}}^\top\mathbf{x}>0~\forall i=1,2,\dots,n
$$