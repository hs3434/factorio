有一张图，有m个item、n个formula两种类型节点，同种节点不互相连接，设节点的权重为$\alpha$和$\beta$，边的权重为$W_{m\times n}$。

现设想一种传播途径，从某些item节点开始，初始化权重为向量$\alpha_{m\times 1}$。

假设$item_a$权重为$\alpha_a$连接n个formula节点，权重为向量$w_{a|n}$，由于每个item可以由不同的formula输出，所以配置可训练向量$a_{a|n}$映射到概率向量$b_{a|n}=softmax(a_{a|n})$来表示每个item的产出需求由不同的formula输出提供的比例，满足$b_{ak}*\alpha_a=ReLU(w_{ak})*\beta_k$，即$\beta_k=\alpha_{ak} \cdot \frac{b_{ak}}{w_{ak}},k\in [1,n]$，同时由于每个formula有多个item产出，formula的实际需求权重应该由产出item中对formula需求最高的项决定，设$B_{m,n}=[b_1,b_2,...,b_m]$，有$\beta_k=max(\alpha \odot \frac{B_{:,k}}{W_{:,k}})$

那么推广到向量$\beta$，有
$$\beta_{n\times 1}= maxcol(\alpha_{m\times 1} \odot \frac{softmax(A_{m\times n}, dim=1)}{ReLU(W_{m\times n})}).$$

但是会有一个问题，即$W_{m\times n}$的元素可能小于等于0，对应的元素是无效的，不应该纳入计算，所以应该引入一个掩码和epsilon，即
$$M_{m\times n}=\mathbb{I}(W_{m\times n}>0) \in \{0,1\}^{m\times n}$$
$$\beta_{n\times 1}= maxcol(\alpha_{m\times 1} \odot \frac{softmax(A_{m\times n}+ (1−M_{m×n})\odot (-1/epsilon), dim=1)}{max(W_{m\times n}, epsilon)}\odot M_{m\times n}).$$
在实际运算中为了防止溢出，这里使用$-1/epsilon$代替$-\inf$，epsilon取一个极小数，比如1e-8。


以上网络进一步传播，由$\Delta\alpha_{m \times 1}=W_{m\times n} \cdot \beta_{n\times 1}$得到
$$\alpha'_{m\times 1}=ReLU(\alpha_{m\times 1} - \Delta\alpha_{m\times 1})$$

另外，item中有一些特殊的节点为源节点$s=\alpha_{m\times 1} \odot Ms_{m \times 1}$，其中$Ms_{m \times 1}$为标示源节点的掩码向量。源节点的需求不应该由formula产出，但是要累计计算资源需求。即
$$s=s+\alpha'_{m\times 1} \odot Ms_{m \times 1}$$
$$$$
$$\alpha''_{m\times 1}=\alpha'_{m\times 1} \odot (1-Ms_{m \times 1})$$

综上所述，对于有m个item、n个formula两种类型节点，同种节点不互相连接的网络，设边的权重矩阵为$W_{m\times n}$，权重矩阵的掩码矩阵为$M_{m\times n}=\mathbb{I}(W_{m\times n}>0) \in \{0,1\}^{m\times n}$，标示源节点的掩码矩阵为$Ms_{m \times 1}\in \{0,1\}^{m\times 1}$，初始化需求矩阵为$\alpha_{0|m\times 1}$，初始化资源需求矩阵为$s^0_{m \times 1}=\alpha^0_{m\times 1} \odot Ms_{m \times 1}$，则定义网络层：
$$\begin{aligned}
layer_1:&\\
&P^1_{m\times n} = M_{m\times n} \odot softmax(A^1_{m\times n}+ (1−M_{m×n})\odot (-1/epsilon), dim=1)\\
&\beta^1_{n\times 1} = maxcol(\alpha^0_{m\times 1} \odot \frac{P^1_{m\times n}}{max(W_{m\times n}, epsilon)})\\
&\alpha'_{m\times 1}=ReLU(\alpha^0_{m\times 1} - W_{m\times n} \cdot \beta^1_{n\times 1})\\
&s^1_{m \times 1}=s^0_{m \times 1}+\alpha'_{m\times 1} \odot Ms_{m \times 1}\\
&\alpha^1_{m\times 1}=\alpha'_{m\times 1} \odot (1-Ms_{m \times 1})\\
......&\\
layer_i:&\\
&P^i_{m\times n} = M_{m\times n} \odot softmax(A^i_{m\times n}+ (1−M_{m×n})\odot (-1/epsilon), dim=1)\\
&\beta^i_{n\times 1} = maxcol(\alpha^{i-1}_{m\times 1} \odot \frac{P^i_{m\times n}}{max(W_{m\times n}, epsilon)})\\
&\alpha'_{m\times 1}=ReLU(\alpha^{i-1}_{m\times 1} - W_{m\times n} \cdot \beta^{i}_{n\times 1})\\
&s^i_{m \times 1}=s^{i-1}_{m \times 1}+\alpha'_{m\times 1} \odot Ms_{m \times 1}\\
&\alpha^i_{m\times 1}=\alpha'_{m\times 1} \odot (1-Ms_{m \times 1})\\

\end{aligned}$$


