import math
import json
import copy
from pathlib import Path


# formula_file = Path("formula.json")
# with open(formula_file, "r") as f:
#     formula = json.load(f, cls=json.JSONDecoder)


def multiply_speed(obj: dict, speed: float) -> dict:
    for key in obj.keys():
        if type(obj[key]) is dict:
            multiply_speed(obj[key], speed)
        elif type(obj[key]) is int or type(obj[key]) is float:
            obj[key] = obj[key] * speed
    return obj


