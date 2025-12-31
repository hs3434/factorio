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
        
        def enter_stack(current_stack):
            
            
        while stack:
            is_processed , current_stack = stack.pop()
            if not is_processed:
                for child_stack in enter_stack(current_stack):
                    stack.append((True, child_stack))
            else:
                exit_stack(current_stack)
        return dp, info    
    def search_path(self, target: str, is_source: Callable[[nx.DiGraph, str], bool] = is_base_type, weight="ratio"):
        g = self.g
        current_path = nx.DiGraph()
        current_path.add_node(target, need=1)
        # stack: is_processed, [current_node, current_path]
        stack: list[tuple[bool, tuple[str, nx.DiGraph, dict]]] = [(False, (target, current_path, dict()))]
        stack2 = []
        path = []
        res_product = -1
        
        def enter_stack(current_stack: tuple[str, nx.DiGraph, dict]):
            current_node, current_path, rest = current_stack
            if g.nodes[current_node].get("type", "") == "formula":
                formula_weight = current_path.nodes[current_node].get("need", 1)
                target = g.nodes[current_node].get("target", "")
                for child, _, data in g.out_edges(current_node, data=True):
                    rest[child] = rest.get(child, 0) + formula_weight * data.get("weight", 1)
                for child, _, data in g.in_edges(current_node, data=True):
                    rest[child] = rest.get(child, 0) - formula_weight / data.get("weight", 1)
                if target in rest and rest[target] > 0:
                    new_path = current_path.copy()
                    new_path.add_node(current_node)
                    new_path.nodes[current_node]["need"] = new_path.nodes[current_node].get("need", 0) + formula_weight
                    for node in rest:
                        if rest[node] > 0:
                            new_path.add_edge(current_node, node)
                            new_path.nodes[node]["need"] = new_path.nodes[node].get("need", 0) - rest[node]
                        else:
                            new_path.add_edge(node, current_node)
                            new_path.nodes[node]["need"] = new_path.nodes[node].get("need", 0) - rest[node]
                            if not is_source(g, node):
                                stack.append((False, (node, new_path, rest)))
            else:
                for parent, _, data in g.in_edges(current_node, data=True):
                    formula_weight = current_path.nodes[current_node]["need"] / data.get(weight, 1)
                    new_path = current_path.copy()
                    new_path.add_edge(parent, current_node)
                    new_path.nodes[parent]["need"] = new_path.nodes[parent].get("need", 0) + formula_weight
                    new_path.nodes[parent]["target"] = current_node
                    stack.append((False, (parent, new_path, rest.copy())))
            return [current_stack]
        
        def exit_stack(current_stack):
            current_node, current_path, rest = current_stack
            if g.nodes[current_node].get("type", "") == "formula":
                pass
            else:
                current_path.add_edge(parent, current_node)
                rest[current_node] = rest.get(current_node, 0) - need
            return current_stack
        
        while stack:
            is_processed , current_stack = stack.pop()
            if not is_processed:
                for child_stack in enter_stack(current_stack):
                    stack.append((True, child_stack))
            else:
                exit_stack(current_stack)
                
       
formula = FormulaDiGraph()
