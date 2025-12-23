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
        while heap:
            curr_min_reciprocal, node = heapq.heappop(heap)
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
        
formula = FormulaDiGraph()
