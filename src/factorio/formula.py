#!/bin/env python3
from __future__ import annotations
import copy
import json
import heapq
import networkx as nx
from pathlib import Path
from collections import OrderedDict
from util import multiply_speed

formula_file = Path("/home/bio-24/projects/factorio/src/formula.json")

with open(formula_file, "r") as f:
    formula = json.load(f, cls=json.JSONDecoder)


def caculate_ratio(obj, src):
    pass

class FormulaDiGraph:
    g: nx.DiGraph = None
    
    def __init__(self):
        if self.g is None:
            self.init_digraph()
    
    def init_digraph(self):
        self.g = nx.DiGraph()
        with open(formula_file, "r") as f:
            formula = json.load(f, cls=json.JSONDecoder)
        for key in formula:
            input: dict = formula[key]["input"]
            output = formula[key]["output"]
            time = input.pop("time")
            self.g.add_node(key, time=time, type="formula")
            for i in input:
                self.g.add_edge(i, key, ratio=1/input[i])
            for o in output:
                self.g.add_edge(key, o, ratio=output[o])
    
    def max_product(self, target: str, source: str, extra: set=set(), weight="ratio"):
        extra = set(extra)
        extra.add(source)
        G = self.g
        
        visited = {}
        stack = [(False, source, [], 1, 0)]
        stack2 = []
        path = []
        res_product = -1
        while stack:
            (is_processed , current_node, current_path, current_product, res_n) = stack.pop()
            if current_node == target:
                    path = current_path
                    res_product = current_product
                    stack2.append((path, res_product))
            elif not is_processed:
                # if G.nodes[current_node].get("type", "") == "formula":
                #     flag = [pre not in extra for pre in G.predecessors(current_node)]
                #     if any(flag):
                #         path, res_product = [current_node], -1
                #         continue
                successors = list(G.successors(current_node))
                res_n = len(successors)
                stack.append((True, current_node, current_path, current_product, res_n))
                for neighbor in G.successors(current_node):
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
        
print(FormulaDiGraph().max_product("petroleum gas", "crude oil", extra="water"))         
            
# class Formula(FormulaDiGraph):
#     def __init__(self, formula_key, ratio=1):
#         super().__init__()
#         if formula_key is None:
#             raise ValueError("formula_key cannot be None")
#         self.formula = formula[formula_key]
#         self.ratio = ratio
#         self.meta_input = copy.deepcopy(self.formula["input"])
#         self.meta_output = copy.deepcopy(self.formula["output"])
#         self.init_io()
#         self.suffix: OrderedDict[Formula] = OrderedDict()
#         self.prefix: OrderedDict[Formula] = OrderedDict()
    
#     def init_io(self):
#         self.tmp_input = multiply_speed(
#             copy.deepcopy(self.meta_input), self.ratio)
#         self.tmp_ouput = multiply_speed(
#             copy.deepcopy(self.meta_output), self.ratio)

#     def update_io(self, ratio: float):
#         self.tmp_input = multiply_speed(self.tmp_input, ratio)
#         self.tmp_ouput = multiply_speed(self.tmp_ouput, ratio)

#     def bind_suffix(self, suffix: Formula):
#         self.suffix.update({suffix: suffix})
#         suffix.prefix.update({self: self})

#     def unbind_suffix(self, suffix: Formula):
#         self.suffix.pop(suffix)
#         suffix.prefix.pop(self)

#     def bind_prefix(self, prefix: Formula):
#         prefix.suffix.update({self: self})
#         self.prefix.update({prefix: prefix})

#     def unbind_prefix(self, prefix: Formula):
#         prefix.suffix.pop(self)
#         self.prefix.pop(prefix)

#     def caculate_input(self):
#         res_dic = copy.deepcopy(self.tmp_input)
#         for obj in self.prefix:
#             ratio = []
#             for key in obj.tmp_ouput:
#                 if key in self.tmp_input:
#                     ratio.append(self.tmp_input[key] / obj.tmp_ouput[key])
#                     res_dic.pop(key)
#             obj.update_io(max(ratio))
#             obj_dic = obj.caculate_input()
#             for key in obj_dic:
#                 if key in res_dic:
#                     res_dic[key] += obj_dic[key]
#                 else:
#                     res_dic[key] = obj_dic[key]
#         return res_dic

#     def caculate_output(self):
#         res_dic = copy.deepcopy(self.tmp_ouput)
#         for obj in self.suffix:
#             ratio = []
#             for key in obj.tmp_input:
#                 if key in self.tmp_ouput:
#                     ratio.append(self.tmp_ouput[key] / obj.tmp_input[key])
#                     res_dic.pop(key)
#             obj.update_io(min(ratio))
#             obj_dic = obj.caculate_output()
#             for key in obj_dic:
#                 if key in res_dic:
#                     res_dic[key] += obj_dic[key]
#                 else:
#                     res_dic[key] = obj_dic[key]
#         return res_dic

#     def multiply_speed(self, speed=1):
#         self.init_io()
#         time = self.tmp_input["time"]
#         self.tmp_input = multiply_speed(self.tmp_input, speed/time)
#         self.tmp_ouput = multiply_speed(self.tmp_ouput, speed/time)


# yycl = Formula(formula_key="Advanced oil processing")
# zycl = Formula(formula_key="Heavy oil cracking")
# qycl = Formula(formula_key="Light oil cracking")
# yycl.bind_suffix(zycl)
# yycl.bind_suffix(qycl)
# zycl.bind_suffix(qycl)
# yycl.multiply_speed()
# print(yycl.caculate_output())
# qycl.multiply_speed()
# print(qycl.caculate_input())