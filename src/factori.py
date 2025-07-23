#!/bin/env python3
import math
import json
from pathlib import Path


formula_file = Path("formula.json")
with open(formula_file, "r") as f:
    formula = json.load(f, cls=json.JSONDecoder)


def multiply_speed(obj: dict, speed: float) -> dict:
    for key in obj.keys():
        if type(obj[key]) is dict:
            multiply_speed(obj[key], speed)
        elif type(obj[key]) is int or type(obj[key]) is float:
            obj[key] = obj[key] * speed
    return obj


class MetaFactorial:
    def __init__(self, key=None, speed: float = 1, x_a: int = 1, y_a: int = 1,
                 x_b: int = 0, y_b: int = 0) -> None:
        self.x_a = x_a
        self.x_b = x_b
        self.y_a = y_a
        self.y_b = y_b
        self.formula = formula[key]
        self.speed = speed
        self.meta_input = multiply_speed(
            formula[key]["input"], speed/formula[key]["input"]["time"])
        self.meta_output = multiply_speed(
            formula[key]["output"], speed/formula[key]["input"]["time"])
        self.input = self.meta_input
        self.output = self.meta_output

    def deploy_area(self, x: int = 1, y: int = 1) -> int:
        self.x = x
        self.y = y
        self.area = self.x * self.y
        self.x_n = math.floor((x - self.x_b) / self.x_a)
        self.y_n = math.floor((y - self.y_b) / self.y_a)
        self.x_r = x - (self.x_n * self.x_a) - self.x_b
        self.y_r = y - (self.y_n * self.y_a) - self.y_b
        self.count = self.x_n * self.y_n
        self.count_io()
        return self.count

    def deploy_xcount(self, count: int = 1, x: int = 1, area_ceil=True) -> int:
        self.count = count
        self.x = x
        self.x_n = math.floor((x - self.x_b) / self.x_a)
        self.x_r = x - (self.x_n * self.x_a) - self.x_b
        self.y_r = 0
        self.y_n = math.ceil(self.count / self.x_n)
        self.y = self.y_n * self.y_a + self.y_b
        self.area = self.x * self.y
        if not area_ceil:
            rest = self.x_n * self.y_n - self.count
            self.area -= rest*self.x_a*self.y_a
        self.count_io()
        return self.y

    def deploy_ycount(self, count: int = 1, y: int = 1, area_ceil=True) -> int:
        self.count = count
        self.y = y
        self.y_n = math.floor((y - self.y_b) / self.y_a)
        self.y_r = y - (self.y_n * self.y_a) - self.y_b
        self.x_r = 0
        self.x_n = math.ceil(self.count / self.y_n)
        self.x = self.x_n * self.x_a + self.x_b
        self.area = self.x * self.y
        if not area_ceil:
            rest = self.x_n * self.y_n - self.count
            self.area -= rest*self.x_a*self.y_a
        self.count_io()
        return self.x

    def count_io(self) -> None:
        self.input = multiply_speed(self.meta_input, self.count)
        self.output = multiply_speed(self.meta_output, self.count)


yycl = MetaFactorial(key="Advanced oil processing", speed=4, x_a=15, x_b=4, y_a=10, y_b=9)


class Factorial(object):
    def __init__(self) -> None:
        pass
