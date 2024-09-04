import numpy as np
import pandas as pd

## q1 

x = np.arange(2, 13)
prob = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]) / 36

Ex = sum(x * prob) # 7

Var = sum((x - Ex) ** 2 * prob) # 5.83

## q2

x_q = Ex * 2 + 3 # 17

std_q =np.sqrt(Var * 4) # 4.83