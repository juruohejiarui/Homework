# Cluster

Given a dataset $\mathcal{D}$ consist of $n$ data points. Seperate it into $K$ clusters: $\Delta=C_1, C_2, \dots, C_K$ , which is a partition of $\mathcal{D}$ .

- $\mathcal{L}(\Delta)$ the cost (loss) of $\Delta$
- $k(i)$ the cluster label of $i$-th data point

## Metric

### External

Measure how weel the clustering results match the groundtruth labels. 聚类结果和答案的相符程度

**Purity**: Measures the fraction of correctly classified points 正确分类的比例

$$
\text{Purity}=\frac{1}{n}\sum_{i=1}^k \max_{j}\left\vert C_i\cap L_j \right\vert
$$

- $C_i=\left\{(\mathbf{x}_a,y_a)~|~k(a)=i,a=1,2,\dots n\right\}$ 类 $i$ 的点集
- $L_i=\left\{(\mathbf{x}_a,y_a)~|~y_a=i,a=1,2,\dots n\right\}$ 答案为 $i$ 的点集

**Rand Index (RI)** : measure pairs of point. 计算点对

$$
\begin{aligned}
\text{RI}&=\frac{\text{TN}+\text{TP}}{\binom{n}{2}}=\frac{\text{TN}+\text{TP}}{\text{TN}+\text{TP}+\text{FN}+\text{FP}}\in [0, 1] \\
\text{TN} &=\sum_{i=1}^n \sum_{j=1}^{i-1}\mathbb{I}(k_i\ne k_j, y_i\ne y_j) \\
\text{TP} &=\sum_{i=1}^n \sum_{j=1}^{i-1}\mathbb{I}(k_i= k_j, y_i= y_j) \\
\text{FN} &=\sum_{i=1}^n \sum_{j=1}^{i-1}\mathbb{I}(k_i\ne k_j, y_i=y_j) \\
\text{FP} &=\sum_{i=1}^n \sum_{j=1}^{i-1}\mathbb{I}(k_i= k_j, y_i\ne y_j) \\
\end{aligned}
$$

higher $\text{RI}\rightarrow$ better match.

**Adjusted Rand Index (ARI)** : adjusted to handle random clustering

$$
\begin{aligned}
\text{ARI} &= \frac{\text{RI}-\mathbb{E}[\text{RI}]}{\max(\text{RI})-\mathbb{E}[\text{RI}]} \\
&=\frac{\sum_{i,j}\binom{n_{i,j}}{2}-\frac{1}{2}\left(\sum_i \binom{|L_i|}{2}\sum_j\binom{|C_j|}{2}\right)}{\frac{1}{2}\left(\sum_i \binom{|L_i|}{2}+\sum_j \binom{|C_i|}{j}\right)-\frac{1}{2}\left(\sum_i \binom{|L_i|}{2}\sum_j\binom{|C_j|}{2}\right)} \\
\text{where } n_{i,j} &= \left\vert L_i\cap C_j \right\vert
\end{aligned}
$$

**Fowlkes-Mallows Index (FMI)** : geometric mean of precision and recall for pairs of points

$$
\begin{aligned}
\text{FMI} &= \sqrt{\text{Precision}\cdot\text{Recall}} \\
\text{Precision}&=\frac{\text{TP}}{\text{TP}+\text{FP}} \\
\text{Recall}&=\frac{\text{TP}}{\text{TP}+\text{FN}}
\end{aligned}
$$

**Normalized Mutual Information (NMI)**

$$
\begin{aligned}
\text{NMI} &= \frac{2\cdot I(C,L)}{H(C)+H(L)} \in [0, 1] \\
\end{aligned}
$$

**V-Measure**: harmonic mean of two complementary criteria: homogeneity and completeness

- **Homogeneity**: purity of cluster $H=1-\frac{H(C|L)}{H(C)}$
- **Completeness**: purity of label $C=1-\frac{H(L|C)}{H(L)}$

$$
\text{V-measure}=2\cdot \frac{H\cdot C}{H+C} \in [0, 1]
$$

### Internal

Assess the goodness of clusters bases only on the intrinsic structure of the data.

- Compactness of clusters （同一类的聚拢程度）
- Separation between clusters （不同类之间的分隔程度）

**Silhouette Score**: how similar a points is to its own cluster compared to other clusters

$$
\begin{aligned}
s(i)&=\frac{b(i)-a(i)}{\max\{a(i), b(i)\}}=\begin{cases}
1 - a(i)/b(i) & \text{if } a(i)<b(i) \\
-1 + b(i)/a(i) & \text{if } a(i)\ge b(i)
\end{cases} \in [-1, 1] \\
a(i) &= \frac{1}{|C_{k(i)}|-1}\sum_{j\in C_{k(i)}} d(i,j) \text{ the average distance to other points in the same cluster} \\
b(i) &= \min_{k\ne k(i)}\frac{1}{|C_k|}\sum_{j\in C_k} d(i,j) \text{ the average distance to points in the nearest cluster} \\
\end{aligned}
$$

**Calinski-Harabasz Index (CH)**: ratio of the variance between clusters to the variance within clusters 类之间的方差和类之间方差的比率

$$
\begin{aligned}
\text{CH} &= \frac{\frac{1}{K-1}\sum_{i=1}^K |C_i|\left\Vert\mathbf{c}_i-\mathbf{c}\right\Vert^2}{\frac{1}{n-K}\sum_{i=1}^K \sum_{x\in C_i}\left\Vert\mathbf{x}-\mathbf{c}_i\right\Vert^2} \\
\end{aligned}
$$

$\mathbf{c}_i$ is the centroid of cluster $C_i$ 第 $i$ 类的中心点（一个数据点）, $\mathbf{c}$ is the centroid of all data points. 所有数据点的中点

**Davies-Bouldin Index (DB)**: average similarity between each cluster and its most similar cluster

$$
\begin{aligned}
\text{DB} &= \frac{1}{K}\sum_{i=1}^K \max_{j\ne i}\left\{\frac{S_i+S_j}{d(c_i,c_j)}\right\} \\
S_i &= \frac{1}{|C_i|}\sum_{x\in C_i}\left\Vert\mathbf{x}-\mathbf{c}_i\right\Vert^2 \\
&\text{ the average distance of points in cluster } C_i \text{ to its centroid } \mathbf{c}_i \\
&\text{类} i \text{中的点与其中心点的距离的平均距离}\\
d(c_i,c_j) &= \left\Vert\mathbf{c}_i-\mathbf{c}_j\right\Vert \\
&\text{ the distance between centroids of clusters } C_i \text{ and } C_j \\
&\text{类 }i \text{ 和类 } j \text{的中心点之间的距离} \\
\end{aligned}
$$

## Methods

### Hard Partitioning Clustering

A data point belongs to only one cluster. 每个数据点只能属于一个类。

#### K-Means

cost (quadratic distortion):

$$
\mathcal{L}(\Delta)=\sum_{i=1}^K \sum_{\mathbf{x}\in C_i}\left\Vert\mathbf{x}-\mathbf{c}_i\right\Vert^2
$$

Proposition: decreases at every step.

Algorithm:

$$
\begin{aligned}
&\textbf{Input: } \mathcal{D}=\{\mathbf{x}_1,\mathbf{x}_2,\dots,\mathbf{x}_n\}, K \\
&1. \text{Initialize } K \text{centers } \mathbf{c}_1, \mathbf{c}_2, \dots, \mathbf{c}_K \text{ randomly} \\
&2. \textbf{Repeat} \\
&3. \text{~~} \textbf{for } i=1 \text{ to } n \textbf{ do} \\
&4. \text{~~~~} k(i)\leftarrow\arg\min_{k=1}^K \left\Vert\mathbf{x}_i-\mathbf{c}_k\right\Vert^2 \\
&5. \text{~~} \textbf{end for} \\
&6. \text{~~} \textbf{for } k=1 \text{ to } K \textbf{ do} \\
&7. \text{~~~~} \mathbf{c}_k\leftarrow\frac{1}{|C_k|}\sum_{\mathbf{x}\in C_k}\mathbf{x} \text{ update the center of cluster } C_k \\
&8. \text{~~} \textbf{end for} \\
&9. \textbf{Until} \text{ meet stop condition} 
\end{aligned}
$$

PS: $\mathbf{c}_k$ in this algorithm is different from the centroid $\mathbf{c}_i$ in the metric section. The former is a center point of a cluster, while the latter is the average of all points in a cluster. 该算法中的中心点可能不是数据点。

人话来说就是首先随机生成 $K$ 个中心点，然后进行循环直到触发结束条件。每次循环将一个点归类到最近的中心点，然后更新中心点为该类所有点的平均值。

Stop conditions:

1. Meet maximum iterations 达到最大迭代次数
2. Assignments no longer change $k(i)$ 不再变化
3. convergence to loca loptimum of cost function $\mathcal{L}(\Delta)$ 成本函数不再变化

PS: 初始化的时候不能随机选取 $K$ 个“数据点” The probability of hitting all $K$ clusters with $K$ samples approaches $0$ as $n$ increases. 选择 $K$ 个数据点的概率随着 $n$ 的增加而趋近于 $0$。



### Soft Partitioning Clustering

A data point can belong to multiple clusters with different degrees of membership. 每个数据点可以属于多个类，且每个类的隶属度不同。

$\gamma_{k,i}$ is the degree of membership of data point $i$ to cluster $k$. 隶属度

$$
\sum_{k=1}^K \gamma_{k,i}=1 \quad \forall~ i=1,2,\dots,n
$$