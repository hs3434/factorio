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

with open(formula_path, "r") as f:
    formula = json.load(f, cls=json.JSONDecoder)

priority = {
    "水": 0
}

def is_base_type(g: nx.DiGraph, node: str) -> bool:
    base_info = g.nodes[node].get("base_info", None)
    if base_info is None:
        return False
    else:
        return base_info["类别"] == "矿石" or base_info["name"] == "水"

    
class FormulaDiGraph:
    g: nx.DiGraph
    
    def __init__(self):
        self.formula_path = formula_path
        if getattr(self, "g", None) is None:
            self.init_digraph()
    
    def init_digraph(self):
        g = nx.DiGraph()
        self.g = g
        with open(self.formula_path, "r") as f:
            formula = json.load(f, cls=json.JSONDecoder)
        for key in formula:
            self.g.add_node(key, type="item", base_info=formula[key]["base_info"])
        for key in formula:
            formula_list = formula[key]["formula_list"]
            for f in formula_list:
                name = f["name"] + "-formula"
                if name not in self.g:
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
    
    def min_inputs(self, target: str, is_source: Callable[[nx.DiGraph, str], bool] = is_base_type, weight="ratio"):
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
        return dp, info    
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
