import numpy as np
from itertools import product

def p_to_m(num:int)->int:
    """Converts p (0, 1, 2, 3,...) to m (0, -1, 1, -2,...)"""
    return int(num / 2) if num % 2 == 0 else int(-((num + 1) / 2))

def m_to_p(number:int)->int:
    """Converts m (0, -1, 1, -2,...) to p (0, 1, 2, 3,...)"""
    return 2 * abs(number + 1) + 1 if number < 0 else 2 * number

def state_generator(state_total:int, site_total:int):
    """Generates all combinations of states for multiple sites"""
    return product(range(state_total), repeat = site_total)

def str_to_bool(s):
    """Converts str True or False to bool True or False"""
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError("Could not convert to bool")