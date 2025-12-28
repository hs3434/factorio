#!/bin/env python3
from __future__ import annotations
import copy
import json
import heapq
import networkx as nx
from pathlib import Path
from collections import OrderedDict, defaultdict
from util import multiply_speed
from typing import Any, Dict, List, Callable

from .spider import formula_path


priority = {
    "水": 0
}

def is_base_type(g: nx.DiGraph, node: str) -> bool:
    base_info = g.nodes[node].get("base_info", None)
    if base_info is None:
        return False
    else:
        return base_info["类别"] == "矿石" or base_info["name"] == "水"

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

# -------------------------- 第二步：保留合理环的配方GNN --------------------------
class RecipeGNN(nn.Module):
    def __init__(
        self,
        G: nx.DiGraph, target_item: str,
    ):
        super(RecipeGNN, self).__init__()
        # 加载图解析结果
        self.G = G
        self.target_item = target_item
        self.parse_recipe_graph()
        # 1. 配方权重（固定）
        self.fixed_weight_matrix = torch.tensor(self.weight_matrix, dtype=torch.float32, requires_grad=False)

        # 2. 配方节点权重（可训练，学习配方优先级）
        self.formula_weights = nn.Parameter(torch.full([self.formula_num], fill_value=1, dtype=torch.float32))

        self.resource_change = torch.full([self.item_num], fill_value=0, dtype=torch.float32)

    def parse_recipe_graph(
            self,
            is_source: Callable[[nx.DiGraph, str], bool] = is_base_type,
            weight="speed"
        ) -> None:
        """解析nx.DiGraph配方图，提取核心信息（复用，新增循环相关标记）"""
        G = self.G
        target_item = self.target_item
        item_nodes = set([n for n in G.nodes if G.nodes[n]['type'] == 'item'])
        formula_nodes = set([n for n in G.nodes if G.nodes[n]['type'] == 'formula'])
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
                    node_idx = item2idx[node]
                    weight_matrix[node_idx, idx] = rest[node]

        self.base_idxs = [item2idx[base] for base in base_resources]
        self.target_idx = item2idx[target_item]
        
        self.weight_matrix = weight_matrix
        self.formula_num = len(formula_nodes)
        self.item_num = len(item_nodes)
        self.non_base_idx = non_base_idx
    
    def forward(self) -> torch.Tensor:
        """
        获取各资源单位时间增减数量张量
        """
        self.resource_change = self.fixed_weight_matrix @ self.formula_weights
        return self.resource_change

    def recipe_loss_fn(
        self,
        resource_change: torch.Tensor,
        target_base_ratio: torch.Tensor,  # 基础资源的目标配比
        lambda_target: float = 10.0,      # 目标物料的损失权重（优先级最高）
        lambda_base_ratio: float = 5.0,   # 基础资源配比的损失权重
        lambda_base_max: float = 1.0,     # 基础资源最大化的损失权重
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
        # -------------------------- 损失项1：目标物料趋近1（核心目标） --------------------------
        target_change = resource_change[self.target_idx]
        loss_target = torch.nn.functional.mse_loss(target_change, torch.tensor(target_num, dtype=torch.float32))

        # -------------------------- 损失项2：基础资源尽可能大 --------------------------
        # 提取基础资源的增减量 [len(base_resources),]
        base_change = resource_change[self.base_idxs]
        # 目标：最大化 base_change → 损失函数中最小化 (-base_change) 的均值（越大，损失越小）
        loss_base_max = -torch.mean(base_change)  # 若base_change为负，-base_change为正，均值越大表示base_change越小（惩罚）

        # -------------------------- 损失项3：基础资源接近给定配比 --------------------------
        # 对基础资源增减量做归一化（得到相对比例），再与目标配比计算MSE
        # 防止除0：添加微小epsilon
        base_change_sum = torch.clamp(torch.sum(torch.abs(base_change)), min=1e-8)
        base_change_ratio = torch.abs(base_change) / base_change_sum  # 归一化后的配比（非负）
        # 确保 target_base_ratio 形状匹配
        loss_base_ratio = torch.nn.functional.mse_loss(base_change_ratio, target_base_ratio)

        # -------------------------- 损失项4：其他资源大于0（非基础、非目标） --------------------------
        # 提取其他资源的索引：非基础 + 非目标

        other_change = self.resource_change[self.non_base_idx]
        
        # 惩罚小于0的部分：使用ReLU的反向逻辑（若x<0，惩罚为-x；x≥0，惩罚为0）
        loss_other_pos = torch.mean(torch.nn.functional.relu(-other_change))  # other_change<0时，-other_change>0，产生惩罚

        # -------------------------- 总损失：加权组合 --------------------------
        total_loss = (
            lambda_target * loss_target
            + lambda_base_ratio * loss_base_ratio
            + lambda_base_max * loss_base_max
            + lambda_other_pos * loss_other_pos
        )

        # 返回总损失及各分项损失（方便监控）
        return total_loss, {
            'loss_target': loss_target.item(),
            'loss_base_max': loss_base_max.item(),
            'loss_base_ratio': loss_base_ratio.item(),
            'loss_other_pos': loss_other_pos.item()
        }

    def train_recipe_gnn(self,
        target_base_ratio: np.ndarray,
        epochs: int = 1000,
        lr: float = 0.01,
        print_interval: int = 100
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

        # 2. 转换目标配比为torch张量（确保形状匹配）
        target_base_ratio_tensor = torch.tensor(target_base_ratio, dtype=torch.float32)
        assert len(target_base_ratio_tensor) == len(model.base_idxs), \
            f"目标配比长度{len(target_base_ratio_tensor)}与基础资源数量{len(model.base_idxs)}不匹配"

        # 3. 训练循环
        model.train()  # 切换到训练模式
        for epoch in range(epochs):
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播：获取资源增减量
            resource_change = model()

            # 计算损失
            total_loss, loss_details = recipe_loss_fn(
                model=model,
                resource_change=resource_change,
                target_base_ratio=target_base_ratio_tensor
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
                print(f"  基础资源配比损失：{loss_details['loss_base_ratio']:.4f}")
                print(f"  其他资源非负损失：{loss_details['loss_other_pos']:.4f}")
                # 打印关键指标
                target_change = resource_change[model.target_idx].item()
                base_changes = resource_change[model.base_idxs].detach().numpy()
                print(f"  目标物料增减量：{target_change:.4f}（目标：1.0）")
                print(f"  基础资源增减量：{base_changes}")
                print(f"  基础资源当前配比：{base_changes / np.sum(np.abs(base_changes)):.4f}")
                print("-" * 50)

        print("训练完成！")
        # 输出最终配方权重
        final_formula_weights = model.formula_weights.detach().numpy()
        formula2idx = model.formula2idx
        idx2formula = {v: k for k, v in formula2idx.items()}
        print("\n最终配方权重：")
        for idx in range(len(final_formula_weights)):
            formula_name = idx2formula[idx]
            weight = final_formula_weights[idx]
            print(f"  {formula_name}: {weight:.4f}")

# -------------------------- 测试代码（可验证运行） --------------------------
if __name__ == "__main__":
    # 1. 构建示例配方图
    G = nx.DiGraph()
    # 添加物料节点
    item_list = ["A", "B", "C", "D", "马达"]  # 马达：目标节点；A/B：基础资源；C/D：非基础物料
    for item in item_list:
        is_base = item in ["A", "B"]
        G.add_node(item, type="item", is_base=is_base)
    # 添加配方节点
    formula_list = ["F1", "F2", "F3"]
    for formula in formula_list:
        G.add_node(formula, type="formula")
    # 添加有向边（带weight）
    # F1: A + B → C（输入A/B，输出C）
    G.add_edge("A", "F1", speed=1.0)
    G.add_edge("B", "F1", speed=1.0)
    G.add_edge("F1", "C", speed=2.0)
    # F2: C → D（输入C，输出D）
    G.add_edge("C", "F2", speed=1.0)
    G.add_edge("F2", "D", speed=1.0)
    # F3: D + B → 马达 + A（输入D/B，输出马达/A，形成环）
    G.add_edge("D", "F3", speed=1.0)
    G.add_edge("B", "F3", speed=1.0)
    G.add_edge("F3", "马达", speed=1.0)
    G.add_edge("F3", "A", speed=0.5)

    # 2. 解析配方图
    target_item = "马达"
    parse_result = parse_recipe_graph(G, target_item=target_item, is_source=is_base_type)

    # 3. 初始化模型
    model = ReasonableCycleRecipeGNN(parse_result)

    # 4. 定义基础资源目标配比（A:0.4，B:0.6）
    target_base_ratio = np.array([0.4, 0.6])  # 对应基础资源A、B

    # 5. 训练模型
    train_recipe_gnn(model, target_base_ratio, epochs=2000, lr=0.005, print_interval=200)