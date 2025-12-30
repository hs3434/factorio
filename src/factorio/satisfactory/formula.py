#!/bin/env python3
from __future__ import annotations
import copy
import json
import heapq
import networkx as nx
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Callable

from .spider import formula_path


priority = {
    "水": 0
}

def is_base_type(g: nx.DiGraph, node: str) -> bool:
    base_info = g.nodes[node].get("base_info", None)
    base_type = {'SAM物质', '水', '煤', '石灰石', '硫磺', '粗石英', '钦金矿石', '铀', '铁矿石', '铜矿石', '铝土矿', "原油", "氮气"}
    return base_info is not None and base_info.get("name", "") in base_type

def stack_recursion_process(current_stack: list, enter_stack: Callable[[list], list[list]], exit_stack: Callable[[list], None] ):
    stack: list[tuple[bool, list]] = [(False, current_stack)]
    while stack:
        is_processed , current_stack = stack.pop()
        if is_processed:
            exit_stack(current_stack)
        else:
            for child in enter_stack(current_stack):
                stack.append((True, child))


def identify_cyclic_items(G: nx.DiGraph, item_nodes: List[str]) -> Set[str]:
    """识别参与循环的物料节点（非基础资源，存在往返路径）"""
    cyclic_items = set()
    for item in item_nodes:
        # 检查是否存在从item出发，最终返回item的路径（即参与循环）
        try:
            for path in nx.all_simple_paths(G, item, item):
                if len(path) > 1:
                    cyclic_items.add(path)
                    break
        except:
            continue
    return cyclic_items


class FormulaDiGraph:   
    g: nx.DiGraph
    
    def __init__(self):
        self.formula_path = formula_path
        if getattr(self, "g", None) is None:
            self.init_digraph()
    
    def init_digraph(self):
        g = nx.DiGraph()
        self.g = g
    
        with open(self.formula_path, "r", encoding="utf-8") as f:
            formula = json.load(f, cls=json.JSONDecoder)
        for key in formula:
            self.g.add_node(key, type="item", base_info=formula[key]["base_info"])
        for key in formula:
            formula_list = formula[key]["formula_list"]
            for f in formula_list:
                name = f["name"] + "-formula"
                if name not in self.g and f["by"] not in {"转化站", "灌装站"}:
                    self.g.add_node(name, type="formula", sub_type=f["type"], by=f["by"])
                    for i in f["inputs"]:
                        speed = f["inputs"][i]["speed"]
                        if i in f["outputs"]:
                            speed = speed - f["outputs"][i]["speed"]
                        if speed > 0:
                            self.g.add_edge(i, name, speed=speed, ratio=1/speed)
                    for i in f["outputs"]:
                        speed = f["outputs"][i]["speed"]
                        if i in f["inputs"]:
                            speed = speed - f["inputs"][i]["speed"]
                        if speed > 0:
                            self.g.add_edge(name, i, speed=speed, ratio=speed)
    
    def min_inputs(self, target: str, is_source: Callable[[nx.DiGraph, str], bool] = is_base_type, weight="speed"):
        g = self.g
        dp = defaultdict(lambda: float("inf"))
        dp[target] = 1
        heap = [(dp[target], target)]
        result = {}
        need = {}
        visited = set()
        stack = [(0, 1, target)]
        while heap:
            curr_num, node = heapq.heappop(heap)
            visited.add(node)
             
            # 遍历父节点（直接用预构建的映射）
            for parent, _, data in g.in_edges(node, data=True):
                new_reciprocal = curr_min_reciprocal * (1/data[weight])
                if new_reciprocal < dp[parent]:
                    dp[parent] = new_reciprocal
                    info[parent] = {"suf": node, "cycle": ""}
                    if parent in visited:
                        info[parent]["cycle"] = node
                        continue
                    if is_source(g, parent):
                        continue
                    heapq.heappush(heap, (new_reciprocal, parent))
        def enter_stack(current_stack):
            current_node, current_path, rest = current_stack
            return [current_stack]
        
        def exit_stack(current_stack):
            pass
        stack_recursion_process(stack, enter_stack=enter_stack, exit_stack=exit_stack)

     
    def search_path(self, target: str, is_source: Callable[[nx.DiGraph, str], bool] = is_base_type, weight="ratio"):
        g = self.g
        current_path = nx.DiGraph()
        # stack: is_processed, [current_node, current_path]
        stack: list[tuple[bool, list]] = [(False, [target, current_path, dict()])]
        stack2 = []
        path = []
        res_product = -1
        while stack:
            is_processed , current_stack = stack.pop()
            current_node, current_path, rest = current_stack
            if is_source(g, current_node):
                    path = current_path
                    res_product = current_product
                    stack2.append((path, res_product))
            elif not is_processed:
                stack.append((True, current_stack))
                current_path.a
                for neighbor in g.in_edges(current_node):
                    edge_weight = G[current_node][neighbor].get(weight, 1.0)
                    product = current_product * edge_weight
                    stack.append((False, neighbor, [neighbor], product, 0)) 
            else:
                if G.nodes[current_node].get("type", "") == "formula":
                    visited[current_node] = visited.get(current_node, 0) + 1
                    tmp_path, tmp_product = [], 0
                    while res_n > 0:
                        res_n -= 1
                        (path, res_product) = stack2.pop()
                        if res_product > 0:
                            tmp_path.append(path)
                            tmp_product += res_product
                    if len(tmp_path) > 1:
                        current_path.extend(tmp_path)
                    elif len(tmp_path) == 1:
                        current_path.extend(tmp_path[0])
                    stack2.append((current_path, tmp_product))
                else:
                    tmp_path, tmp_product = [], -1
                    while res_n > 0:
                        res_n -= 1
                        (path, res_product) = stack2.pop()
                        if res_product > tmp_product:
                            tmp_product = res_product
                            tmp_path = path
                    current_path.extend(tmp_path)
                    stack2.append((current_path, tmp_product))
        return stack2.pop()
       
formula = FormulaDiGraph()


import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Set
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备：{device}")

# -------------------------- 第二步：保留合理环的配方GNN --------------------------
class RecipeGNN(nn.Module):
    def __init__(
        self,
        G: nx.DiGraph, target_item: str, device=device
    ):
        self.device = device
        super(RecipeGNN, self).__init__()
        self.G = G
        self.target_item = target_item
        self.parse_recipe_graph()
        # 1. 固定权重矩阵
        self.fixed_weight_matrix = torch.tensor(self.weight_matrix, dtype=torch.float32, requires_grad=False, device=self.device)
        # 2. 可训练配方权重（初始值为1，无约束）
        self.formula_weights = nn.Parameter(torch.full([self.formula_num], fill_value=1, dtype=torch.float32, device=self.device))
        # 3. 定义非负激活函数（Softplus 平滑非负，推荐；也可使用 ReLU）
        # nn.Softplus(beta=10)：推荐，输出为 ln(1 + e^(beta*x))，平滑非负，梯度始终存在，训练更稳定。
        # nn.ReLU()：备选，输出 max(0, x)，简单高效，但当 x<=0 时梯度为 0，可能导致权重无法更新。
        self.non_neg_activation = nn.Softplus(beta=10)  # beta越大，越接近 ReLU
        # self.non_neg_activation = nn.ReLU()  # 备选，简单但在0点梯度为0
        self.init_train_weight(base_weight={})

    def parse_recipe_graph(
            self,
            is_source: Callable[[nx.DiGraph, str], bool] = is_base_type,
            weight="speed"
        ) -> None:
        """解析nx.DiGraph配方图，提取核心信息（复用，新增循环相关标记）"""
        G = self.G
        target_item = self.target_item
        item_nodes = set([n for n in G.nodes if G.nodes[n].get('type', "item") == 'item'])
        formula_nodes = set([n for n in G.nodes if G.nodes[n].get('type', "item") == 'formula'])
        base_resources = set([n for n in item_nodes if is_source(G, n)])

        if target_item not in item_nodes:
            raise ValueError(f"目标节点{target_item}不是物料节点")
        if target_item in base_resources:
            raise ValueError(f"目标节点{target_item}是基础资源")

        item2idx = {n: i for i, n in enumerate(item_nodes)}
        formula2idx = {n: i for i, n in enumerate(formula_nodes)}
        idx2item = {i: n for n, i in item2idx.items()}
        idx2formula = {i: n for n, i in formula2idx.items()}
        non_base_items = set([n for n in item_nodes if n not in base_resources])
        non_base_idx = [item2idx[item] for item in non_base_items]
        weight_matrix = np.zeros((len(item_nodes), len(formula_nodes)), dtype=np.float32)
                    
        for formula in formula_nodes:
            rest = {}
            idx = formula2idx[formula]
            for node, _, data in G.in_edges(formula, data=True):
                rest[node] = rest.get(node, 0) -  data.get(weight, 0)
            for _, node, data in G.out_edges(formula, data=True):
                rest[node] = rest.get(node, 0) +  data.get(weight, 0)
            for node in rest:
                if node in item2idx:
                    node_idx = item2idx[node]
                    weight_matrix[node_idx, idx] = rest[node]

        self.base_resources = base_resources
        self.base_idxs = [item2idx[base] for base in base_resources]
        self.target_idx = item2idx[target_item]
        self.idx2formula = idx2formula
        self.idx2item = idx2item
        
        self.weight_matrix = weight_matrix
        self.formula_num = len(formula_nodes)
        self.item_num = len(item_nodes)
        self.non_base_idx = non_base_idx
    
    def forward(self) -> torch.Tensor:
        """
        获取各资源单位时间增减数量张量
        """
        self.non_neg_formula_weights = self.non_neg_activation(self.formula_weights)
        self.resource_change = self.fixed_weight_matrix @ self.non_neg_formula_weights
        return self.resource_change

    def recipe_loss_fn(
        self,
        lambda_target: float = 10.0,      # 目标物料的损失权重（优先级最高）
        lambda_base_max: float = 0.01,     # 基础资源最大化的损失权重
        lambda_other_pos: float = 2.0,     # 其他资源大于0的损失权重
        target_num = 1
    ) -> tuple[torch.Tensor, dict]:
        """
        多目标损失函数设计
        参数：
            model: 配方GNN模型
            resource_change: 模型输出的资源增减量 [item_num,]
            target_base_ratio: 基础资源的目标配比 [len(base_resources),]
            lambda_*: 各损失项的加权系数（可根据业务调整优先级）
        返回：
            total_loss: 总损失（可反向传播优化）
        """
        resource_change = self.resource_change
        # -------------------------- 损失项1：目标物料趋近target_num（核心目标） --------------------------
        target_change = resource_change[self.target_idx]
        loss_target = torch.nn.functional.mse_loss(target_change, torch.tensor(target_num, dtype=torch.float32).to(device))

        # -------------------------- 损失项2：基础资源尽可能消耗少 --------------------------
        # 提取基础资源的增减量 [len(base_resources),]
        base_change = resource_change[self.base_idxs]
        mask = (base_change < 0).float()
        # 目标：因为base_change为负值表示消耗，所以应该最大化 base_change → 损失函数中最小化 (-base_change) 的均值（越大，损失越小）
        loss_base_max = -torch.mean(mask * base_change * self.scaled_base_weight)  # 若base_change为负，-base_change为正，均值越大表示base_change越小（惩罚）

        # -------------------------- 损失项3：其他资源大于0（非基础、非目标） --------------------------
        # 提取其他资源的索引：非基础 + 非目标

        other_change = self.resource_change[self.non_base_idx]
        
        # 惩罚小于0的部分：使用ReLU的反向逻辑（若x<0，惩罚为-x；x≥0，惩罚为0）
        loss_other_pos = torch.max(torch.nn.functional.relu(-other_change))  # other_change<0时，-other_change>0，产生惩罚

        # -------------------------- 总损失：加权组合 --------------------------
        total_loss = (
            lambda_target * loss_target
            + lambda_base_max * loss_base_max
            + lambda_other_pos * loss_other_pos
        )

        # 返回总损失及各分项损失（方便监控）
        return total_loss, {
            'loss_target': loss_target.cpu().item(),
            'loss_base_max': loss_base_max.cpu().item(),
            'loss_other_pos': loss_other_pos.cpu().item()
        }

    def init_train_weight(self, base_weight: dict = {}):
        min_val, max_val = 0, 1
        target_base_weight = {key: 1 for key in self.base_resources}
        for key, value in base_weight.items():
            if key in self.base_resources:
                target_base_weight[key] = value
        target_base_weight = torch.tensor(list(target_base_weight.values()), dtype=torch.float32)
        # 计算当前权重的最小值和最大值（添加epsilon避免除0）
        weight_min = torch.clamp(torch.min(target_base_weight), min=1e-8)
        weight_max = torch.clamp(torch.max(target_base_weight), min=1e-8)
        # 最小-最大缩放：将权重缩放到 [min_val, max_val]
        scaled_weight = (target_base_weight - weight_min) / (weight_max - weight_min) * (max_val - min_val) + min_val
        self.base_weight = base_weight
        self.scaled_base_weight = scaled_weight.to(device)
            
            
    def train_recipe_gnn(self,
        base_weight: dict,
        epochs: int = 1000,
        lr: float = 0.01,
        print_interval: int = 100,
        lambda_target = 10.0,      # 目标物料的损失权重（优先级最高）
        lambda_base_max = 0.1,     # 基础资源最大化的损失权重
        lambda_other_pos = 10.0,     # 其他资源大于0的损失权重
        target_num = 1
    ) -> None:
        """
        训练配方GNN模型
        参数：
            target_base_ratio: 基础资源的目标配比（numpy数组，长度=基础资源数量）
            epochs: 训练轮数
            lr: 学习率
            print_interval: 日志打印间隔
            """
        # 1. 配置优化器（仅优化可训练参数 formula_weights）
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.init_train_weight(base_weight)
        # 3. 训练循环
        self.train()  # 切换到训练模式
        for epoch in range(epochs):
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播：获取资源增减量
            resource_change = self.forward()

            # 计算损失
            total_loss, loss_details = self.recipe_loss_fn(
                lambda_target = lambda_target,      # 目标物料的损失权重（优先级最高）
                lambda_base_max = lambda_base_max,     # 基础资源最大化的损失权重
                lambda_other_pos = lambda_other_pos,     # 其他资源大于0的损失权重
                target_num = target_num
            )

            # 反向传播 + 参数更新
            total_loss.backward()
            optimizer.step()

            # 打印训练日志
            if (epoch + 1) % print_interval == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  总损失：{total_loss.item():.4f}")
                print(f"  目标物料损失：{loss_details['loss_target']:.4f}")
                print(f"  基础资源最大化损失：{loss_details['loss_base_max']:.4f}")
                print(f"  其他资源非负损失：{loss_details['loss_other_pos']:.4f}")
                # 打印关键指标
                target_change = resource_change[self.target_idx].cpu().item()
                base_changes = resource_change[self.base_idxs].cpu().detach().numpy()
                
                # 避免除0错误
                if np.sum(np.abs(base_changes)) > 1e-8:
                    base_ratio = base_changes / np.sum(np.abs(base_changes))
                else:
                    base_ratio = base_changes
                print(f"  目标物料增减量：{target_change:.4f}（目标：1.0）")
                print(f"  基础资源增减量：{base_changes}")
                print(f"  基础资源当前配比：{base_ratio.round(4)}")
                print("-" * 50)

        print("训练完成！")
        # 输出最终配方权重
        self.final_formula_weights = self.non_neg_formula_weights.cpu().detach().numpy()
        print("\n最终配方权重：")
        for idx in range(len(self.final_formula_weights)):
            formula_name = self.idx2formula[idx]
            weight = self.final_formula_weights[idx]
            print(f"  {formula_name}: {weight:.4f}")

# -------------------------- 测试代码（可验证运行） --------------------------
if __name__ == "__main__":
    from factorio.satisfactory.formula import formula, nx, RecipeGNN
    base_weight={
        "水": 0, "铁矿石": 1/921, "石灰石": 1/693, "铜矿石": 1/369, "钦金矿石": 1/150, "煤": 1/423,
        "原油": 1/126, "硫磺": 1/108, "铝土矿": 1/123, "粗石英": 1/135, "铀": 1/21, "SAM物质": 1/102, "氮气": 1/120
    }
    # 1. 构建示例配方图
    model = RecipeGNN(formula.g, target_item = "镄燃料棒")
    
    model.train_recipe_gnn(
        base_weight=base_weight,
        epochs=20000, lr=0.005, print_interval=200,
        lambda_target=1,
        lambda_base_max=0.01,
        lambda_other_pos=1000
    )
