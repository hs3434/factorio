有一张图，有m个item、n个formula两种类型节点，同种节点不互相连接，设节点的权重为$\alpha$和$\beta$，边的权重为$W_{m\times n}$。

现设想一种传播途径，从某些item节点开始，初始化权重为向量$y_{|m}$。

假设$item_a$权重为$\alpha_a$连接n个formula节点，权重为向量$w_{a|n}$，配置可训练向量$a_{a|n}$，映射到概率向量$b_{a|n}=softmax(a_{a|n})$，满足$b_{ak}*\alpha_a=ReLU(w_{ak})*\beta_k$，即$\beta=\alpha_a \cdot \frac{b_{a}}{w_{a}}$

那么推广到向量$y_{|m}$，有$\beta=\sum^{m}{\alpha_i \cdot \frac{b_{i}}{w_{i}}}$，即
$$B_{n\times 1}= (\frac{softmax(A_{m\times n}, dim=1)}{ReLU(W_{m\times n})})^T \cdot A_{m\times 1}$$
但是会有一个问题，即$W_{m\times n}$的元素可能小于等于0，所以应该引入一个掩码和epsilon，即
$$M_{m\times n}=\mathbb{I}(W_{m\times n}>0) \in {0,1}^{m\times n}$$
$$B_{n\times 1}= (\frac{softmax(A_{m\times n}, dim=1)}{min(W_{m\times n}, epsilon)}\odot M_{m\times n})^{T} \cdot A_{m\times 1}$$
但是这里