import copy
from .util import *


class Assembling_machine:
    def __init__(self, level=2, formula_key=None, transport_belt=30):
        speed_type = {"1": 0.5, "2": 0.75, "3": 1.25}
        self.speed = speed_type[str(level)] if str(level) in speed_type else 0.75
        self.weight = 3
        self.height = 3
        self.transport_belt = 30
        self.init_formula(formula_key)
    
    def deploy_type1(self):
        count = len(self.meta_input)
        if count == 2:
            self.meta_weight = self.weight + 3
            self.meta_height = self.height
        elif count >= 3:
            self.meta_weight = self.weight + count + 2
            self.meta_height = self.height
        
        

    
