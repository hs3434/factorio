# 基于游戏《幸福工厂》的生产网络优化研究
## 需求描述
在游戏《幸福工厂》的生产体系中，配方是如$\sum_{i=1}^{n1} a_i \cdot x_i = \sum_{i=1}^{n1} b_i \cdot y_i$形式的公式，其中$x_i$是输入的原料，$a_i$是单位时间对应原料的消耗量，$y_i$是输出的产物，$b_i$是单位时间对应产物的产出量，一个配方的产出可能是另一个配方的输入，如此构成了生产链路，而链路交叉形成生产网络。
对于网络中的产物，可能存在多种配方，不同的配方有不同的原料消耗和生产效率，而基础原料的产量是有上限的，且丰度不同，于是便有了生产网络优化的需求：使用更少的原料生产更多的目标产品。
不过这个优化目标还是不太明确，我们可以更细化一下：
1. 给定数量的原料，最大化的目标产品的数量，同时对于多个目标产品，应该按照指定配比输出目标产品；
2. 使用更少的原料生产给定数量的目标产品，同样的，应该按照指定配比输入原料。
## 传统线性规划建模
### 方案1
首先，我们对给定数量的原料，最大化的目标产品的数量，同时按照指定配比输出目标产品建模
$$\begin{aligned}
&V \subseteq \{1,2,...,m\} \quad &物品集合\\
&F \subseteq \{1,2,...,n\} \quad &配方集合\\
&T \subseteq \{1,2,...,k\} \quad &目标产品集合\\
&\alpha_i \geq 0, \quad \forall i \in V \quad &初始原料存量，大于0代表有存量\\
&W_{ij} \in \R, \quad \forall i \in V, \forall j\in F \quad &生产权重矩阵，W_{ij}表示每单位配方j对物资i消耗产出量，正数为生产，负数为消耗\\
&\beta_j ≥ 0, \quad \forall j \in F \quad &决策变量：配方j的单位时间运行强度\\
&t_i \in V, \quad \forall i \in T &目标产物索引，其中设 t_1 为基准目标产物索引\\
&k_i > 0,\quad \forall i\in T\quad &各目标产物相对比例\\
max_{\beta}\quad&\sum_{j=1}^{n}W_{t_1j} \cdot \beta_j\\
s.t.\quad&\sum_{j=1}^{n}(-W_{ij}) \cdot \beta_j \leq \alpha_i^0,\quad \forall i \in V\\
    &k_{i} \cdot \sum_{j=1}^{n}W_{t_1j}\cdot \beta_j - \sum_{j=1}^{n}W_{t_ij}\cdot \beta_j = 0, \quad \forall i \in T\\
    &\beta_j \geq 0,\quad \forall j \in  F\\
\end{aligned}$$
代码如下：
```python
import decimal
import numpy as np
from scipy.optimize import linprog
from factorio.satisfactory.formula import read_formula, parse_recipe_graph, parse_item

def print_item_weights(item_weights, idx2item: dict, cut=0.01):
    cut_str = str(cut).rstrip('0').rstrip('.') if '.' in str(cut) else str(cut)
    d = decimal.Decimal(cut_str)
    sign, digits, exponent = d.as_tuple()
    decimal_places = max(0, -int(exponent))
    
    # 步骤3：需要保留的位数 = 有效小数位数 + 1（或直接返回有效小数位数）
    needed_places = decimal_places + 1
    for idx in range(len(item_weights)):
        item_name = idx2item[idx]
        weight = np.round(item_weights[idx], needed_places)
        if np.abs(weight) > cut:
            print(f"  {item_name}: {weight}")

     
     
def satisfactory_factory_optimize(W: np.ndarray, source_weight: np.ndarray, target_weight: np.ndarray):
    """
    《幸福工厂》生产网络优化-线性规划求解（方案1：给定原料，最大化配比产出的目标产品）
    严格适配公式：max β 求和(W[t1,j]*βj)
    约束：1.原料消耗 ≤ 初始存量  2.目标产物严格配比  3.βj≥0
    """

    m, n = W.shape
    target_idx = np.where(target_weight > 0)[0]
    k = target_weight[target_idx]
    # ===================== 2. 线性规划转化：linprog标准型适配 =====================
    # scipy.linprog 标准型：min c·x  s.t.  A_ub·x ≤ b_ub ; A_eq·x = b_eq ; x≥0
    # 原模型是最大化 → 转化为：min (-c)·β ，其中 c_j = W[t1,j]
    c = -W[target_idx[0], :]  # 目标函数系数，取负实现最大化

    # 约束1：原料消耗约束 ∑(-W[i,j]·βj) ≤ α0[i]  → A_ub = -W ， b_ub = alpha0
    A_ub = -W
    b_ub = source_weight
    # 约束2：目标产物配比约束 k_t·∑W[t1,j]βj - ∑W[t,j]βj = 0
    A_eq = np.expand_dims(k, axis=1) * W[[target_idx[0]] * len(k), :] - W[target_idx, :]
    b_eq = np.zeros(len(k))

    # 约束3：配方运行强度非负 βj≥0 → linprog默认约束，无需额外定义
    bounds = [(0, None) for _ in range(n)]

    # ===================== 3. 求解线性规划 =====================
    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs',  # 高效求解器，scipy1.6+推荐
        options={"disp": True}  # 显示求解过程
    )
    return res

# 执行求解
if __name__ == "__main__":
    g = read_formula()
    (
        raw_weight_matrix,
        item_nodes, formula_nodes,
        item2idx, formula2idx,
        idx2item, idx2formula
    ) = parse_recipe_graph(g, weight = "speed")
    source_dict = {
        "水": 1e8, "铁矿石": 921, "石灰石": 693, "铜矿石": 369, "钦金矿石": 150, "煤": 423,
        "原油": 126, "硫磺": 108, "铝土矿": 123, "粗石英": 135, "铀": 21, "SAM物质": 102, "氮气": 120
    }
    
    # 3. 初始化模型

    # source_weight = {k:v * 1e3 for k,v in source_weight.items()}
    target_dict = {
        "封压方块": 1,
        "镄燃料棒": 2
    }
    source_weight = np.zeros(len(item_nodes))
    for key in source_dict:
        source_weight[item2idx[key]] = source_dict[key]
    target_weight = np.zeros(len(item_nodes))
    for key in target_dict:
        target_weight[item2idx[key]] = target_dict[key]
    res = satisfactory_factory_optimize(raw_weight_matrix, source_weight, target_weight)
    if res.success:
        beta_opt = res.x  # 最优配方运行强度 β*
        # 计算各物品的总产出/消耗量
        product_consume = raw_weight_matrix @ beta_opt
        print_item_weights(beta_opt, idx2formula)
        print_item_weights(product_consume, idx2item)
        
    else:
        print(f"求解失败！原因：{res.message}")
```
成功求解。


对于配方生产网络，有m个item、n个formula两种类型节点，同种节点不互相连接。设节点的权重为$\alpha$和$\beta$，代表物资和配方的单位数量，边的权重为$W_{m\times n}$（代表每种配方每单位消耗产出物资数量）。item中有一些特殊的节点为源节点$s=\alpha_{m\times 1} \odot Ms_{m \times 1}$，其中$Ms_{m \times 1}$为标示源节点的掩码向量。源节点的需求不应该由formula产出，但是要累计计算资源需求。现需要对一些目标物资的生产路线建模，优化源节点的物资需求。
### 思路1 产出逆推需求
#### 推导过程
可以设想一种传播途径，从某些item节点开始，初始化权重为向量$\alpha^{pre}_{m\times 1}$，其中大于0的值表示需求，那么对于小于0表示已经提供的物资。对于权重为$\alpha_a>0$的节点$item_a$，需要通过生产公式消耗原料来满足其需求，假设其连接n个formula节点，权重为向量$w_{a|n}$，由于每个item可以由不同的formula输出，所以配置可训练向量$a_{a|n}$映射到概率向量$b_{a|n}=softmax(a_{a|n})$来表示每个item的产出需求由不同的formula输出提供的比例，满足$b_{ak}\cdot \alpha_a=ReLU(w_{ak})\cdot \beta_k$，即$\beta_k=\alpha_{ak} \cdot \frac{b_{ak}}{w_{ak}},k\in [1,n]$，同时由于每个formula有多个item产出，formula的实际需求权重应该由产出item中对formula需求最高的项决定，设$B_{m,n}=[b_1,b_2,...,b_m]$，有$\beta_k=max(\alpha \odot \frac{B_{:,k}}{W_{:,k}})$

那么推广到向量$\beta$，有
$$\begin{aligned}
&\alpha^{pre}_{m\times 1} \in R^{m\times 1} \quad &(当前的物资需求供应量)\\
&W_{m\times n} \in R^{m\times n} \quad &(生产权重矩阵)\\
&Wb_{m\times n}= \frac{1}{W_{m\times n}}\quad &(倒数生产权重矩阵，用于通过物资计算配方需求)\\
&A_{m\times n}\in R^{m\times n} \quad &(配方的分配权重矩阵，被训练参数)\\
&P_{m\times n}=softmax(A_{m\times n}, dim=1)\quad &(配方的分配概率矩阵)\\
&X_{m\times n}=ReLU(\alpha^{pre}_{m\times 1}) \odot P_{m\times n} \odot Wb_{m\times n}\quad &(物资对配方的需求强度矩阵)\\
&\beta_{n\times 1}= maxcol(X) \quad &(配方的需求矩阵)\\
\end{aligned}$$

但是会有一个问题，即$W_{m\times n}$的元素可能小于等于0，表示对应物资不是由对应配方生产，对应元素是无效的，不应该纳入计算，且要防止除0问题，所以应该引入掩码和epsilon，即
$$\begin{aligned}
&M_{m\times n} = \mathbb{I}(W_{m\times n}>0) \in \{0,1\}^{m\times n}\\
&Mp_{m\times n}=(1−M_{m\times n})\odot (-epsilon)\\
&Wb_{m\times n}= \frac{1}{W_{m\times n} + epsilon \odot \mathbb{I}(W_{m\times n}=0)}\\
&P_{m\times n}=softmax(A_{m\times n}+Mp_{m\times n}, dim=1) \odot M_{m\times n}\\
\end{aligned}$$
在实际运算中为了防止溢出，这里使用$-epsilon$代替$-\inf$，epsilon取一个极小数，比如1e-8。

以上网络进一步传播，由$\Delta\alpha_{m \times 1}=W_{m\times n} \cdot \beta_{n\times 1}$得到
$$\alpha'_{m\times 1}=\alpha^{pre}_{m\times 1} - \Delta\alpha_{m\times 1}=\alpha_{m\times 1}-W_{m\times n} \cdot \beta_{n\times 1}$$

另外，item中有一些特殊的节点为源节点$s=\alpha_{m\times 1} \odot Ms_{m \times 1}$，其中$Ms_{m \times 1}$为标示源节点的掩码向量。源节点的需求不应该由formula产出，但是要累计计算资源需求。即
$$s_{m \times 1}=s^{pre}_{m \times 1}+\alpha'_{m\times 1} \odot Ms_{m \times 1}$$
$$\alpha^{suf}_{m\times 1}=\alpha'_{m\times 1} \odot (1-Ms_{m \times 1})$$
#### 归纳建模
综上所述，已知边的权重矩阵为$W_{m\times n}$，标示源节点的掩码矩阵为$Ms_{m \times 1}\in \{0,1\}^{m\times 1}$，初始化需求供应矩阵为$\alpha^0_{m\times 1}$，建模方案如下：
$$\begin{aligned}
init:&\\
&W_{m\times n} \in R^{m\times n} \quad &(生产权重矩阵)\\
&Ms_{m \times 1}\in \{0,1\}^{m\times 1} \quad &(物资的源节点掩码矩阵)\\
&M_{m\times n}=\mathbb{I}(W_{m\times n}>0) \in \{0,1\}^{m\times n}\quad &(生产权重的供应掩码矩阵)\\\
&Ma_{m \times 1}=1-Ms_{m \times 1}\quad &(物资的非源节点掩码矩阵)\\
&Mp_{m\times n}=(1−M_{m\times n})\odot (-epsilon)\quad &(配方的分配概率掩码矩阵)\\
&Wb_{m\times n}= \frac{1}{W_{m\times n} + epsilon \odot \mathbb{I}(W_{m\times n}=0)}\quad &(倒数生产权重矩阵)\\
&\alpha^0_{m\times 1} \in R^{m\times 1} \quad &(初始需求供应矩阵)\\
......&\\
layer^i:&\\
&\alpha^{pre}_{m\times 1}\in R^{m\times 1} \quad &(由上一层的 \alpha^i_{m\times 1} 输入)\\

&A^i_{m\times n}\in R^{m\times n} \quad &(配方的分配权重矩阵，被训练参数)\\
&P^i_{m\times n}=softmax(A^i_{m\times n}+Mp_{m\times n}, dim=1) \odot M_{m\times n} \quad &(配方的分配概率矩阵)\\
&X_{m\times n}=ReLU(\alpha^{pre}_{m\times 1}) \odot P^i_{m\times n} \odot Wb_{m\times n}\quad &(物资对配方的需求强度矩阵)\\
&\beta^i_{n\times 1}= maxcol(X) \quad &(配方的需求矩阵)\\
&\Delta\alpha_{m\times 1}=W_{m\times n} \cdot \beta^i_{n\times 1}\quad &(本层生产处理的的物资变化量)\\
&\alpha^i_{m\times 1}=\alpha^{pre}_{m\times 1} - \Delta\alpha_{m\times 1} \odot Ma_{m \times 1}\quad &(计算改变后的需求供应矩阵)\\
loss:&\\
&Loss=Loss_{source}(s^i_{m \times 1})+Loss_{unsat}(ReLU(−\alpha^i_{m\times 1}))
\end{aligned}$$

#### 复盘总结
针对以上建模，我使用多层网络去训练最优解，其中损失函数由最终&s^i_{m \times 1}$的基础资源的消耗损失和$ReLU(\alpha^i_{m\times 1})$的需求未满足损失，但是发现需求未满足损失无法下降。

总结原因是因为下游产物的需求满足往往会产生更多的上游产物的需求，所以此时在梯度上会趋向不满足下游产物的需求，从而导致需求未满足损失无法下降，另外在满足需求时，也会导致基础资源的消耗损失增加。

### 思路2 资源顺推产出
#### 推导过程

#### 归纳建模
已知边的权重矩阵为$W_{m\times n}$，目标产出的产物比例矩阵$Wa^0_{m\times 1} \in R_+^{m\times 1}$，初始的资源存量矩阵为$\alpha^0_{m \times 1}\in R_+^{m\times 1}$，建模方案如下：
$$\begin{aligned}
init:&\\
&\alpha^0_{m \times 1}\in R_+^{m\times 1} \quad &(初始资源存量矩阵)\\
&Wa_{m\times 1} \in R_+^{m\times 1} \quad &(目标产物的产物权重矩阵)\\

&W_{m\times n} \in R^{m\times n} \quad &(生产权重矩阵)\\
&M_{m\times n}=\mathbb{I}(W_{m\times n}<0) \in \{0,1\}^{m\times n}\quad &(生产权重的需求掩码矩阵)\\
&Mp_{m\times n}=(1−M_{m\times n})\odot (-epsilon)\quad &(配方的分配概率掩码矩阵)\\
&Wb_{m\times n}= \frac{1}{W_{m\times n} + epsilon \odot \mathbb{I}(W_{m\times n}=0)}\quad &(倒数生产权重矩阵)\\

......&\\
layer^i:&\\
&\alpha^{pre}_{m\times 1}\in R_+^{m\times 1} \quad &(由上一层的 \alpha^i_{m\times 1} 输入)\\
&A^i_{m\times n}\in R^{m\times n} \quad &(配方的分配权重矩阵，被训练参数)\\
&P^i_{m\times n}=softmax(A^i_{m\times n}+Mp_{m\times n}, dim=1) \odot M_{m\times n} \quad &(配方的分配概率矩阵)\\
&X_{m\times n}=-ReLU(\alpha^{pre}_{m\times 1}) \odot P^i_{m\times n} \odot Wb_{m\times n}\quad &(物资对配方的供应强度矩阵)\\
&\beta^i_{n\times 1}= mincol(X) \quad &(配方的供应矩阵)\\
&\Delta\alpha_{m\times 1}=W_{m\times n} \cdot \beta^i_{n\times 1}\quad &(本层生产处理的的物资变化量)\\
&\alpha^i_{m\times 1}=\alpha^{pre}_{m\times 1} + \Delta\alpha_{m\times 1}\quad &(计算改变后的资源存量矩阵)\\
loss:&\\
&Loss=Loss_{obj}(-\alpha^i_{m\times 1} \odot Wa_{m\times 1})
\end{aligned}$$