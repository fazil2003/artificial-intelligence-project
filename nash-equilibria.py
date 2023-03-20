import numpy as np
from scipy.optimize import linprog

# Define the payoff matrix
payoff_matrix = np.array([[[[1, -1], [3, -3]], [[0, 0], [2, -2]]], [[[1, -1], [3, -3]], [[0, 0], [2, -2]]]])

# Find the nash equilibria
c = [-1, 1]
A_ub = np.array([[-payoff_matrix[0,0,0,0], -payoff_matrix[0,0,1,0], -payoff_matrix[0,1,0,0], -payoff_matrix[0,1,1,0]],
                 [-payoff_matrix[1,0,0,1], -payoff_matrix[1,0,1,1], -payoff_matrix[1,1,0,1], -payoff_matrix[1,1,1,1]]])
b_ub = np.array([0, 0])
x0_bounds = (0, None)
x1_bounds = (0, None)
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[x0_bounds, x1_bounds])
nash_eq = [1 / res.fun * res.x[0], -1 / res.fun * res.x[1]]

print("Payoff matrix:")
print(payoff_matrix)
print("Nash equilibria:")
print(nash_eq)
