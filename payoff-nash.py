import numpy as np
from scipy.optimize import linprog

# Define the game parameters
g = 0.25
cm1, cm2, ci1, ci2, l, a, b, a0, a1, a2, a3 = 0.15, 0.16, 0.17, 0.48, 0.98, 0.1, 0.2, 0, 0, 0, 0

# Define the player moves
player1_moves = ['R', 'P', 'S']
player2_moves = ['R', 'P', 'S']

# Define the player payoffs for each combination of moves
player1_payoffs = np.array([[[0, 0], [-g, g], [g, -g]],
                            [[g, -g], [0, 0], [-g, g]],
                            [[-g, g], [g, -g], [0, 0]]])

player2_payoffs = -player1_payoffs.transpose((1, 0, 2))

# Find the payoff matrix
payoff_matrix = np.stack((player1_payoffs, player2_payoffs), axis=-1)

print("Payoff Matrix:\n", payoff_matrix)
print(payoff_matrix.shape)

# Find the Nash equilibria
# c = np.array([-1, -1, -1])
# A_ub = np.array([[-payoff_matrix[0,0,0], -payoff_matrix[0,1,0], -payoff_matrix[0,2,0]],
#                  [-payoff_matrix[1,0,1], -payoff_matrix[1,1,1], -payoff_matrix[1,2,1]],
#                  [-payoff_matrix[2,0,0], -payoff_matrix[2,1,1], -payoff_matrix[2,2,1]]])
# b_ub = np.zeros(3)

# nash_eq = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None), (0, None)])

# if nash_eq.success:
#     print("\nNash Equilibria:")
#     for i in range(len(nash_eq.x)):
#         if nash_eq.x[i] > 0:
#             print(player1_moves[i // 3], player2_moves[i % 3])
# else:
#     print("\nNo Nash Equilibria found.")
