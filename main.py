import pandas as pd
import numpy as np
import random

df = pd.read_csv("finaldataset.csv")

# for index, row in df.iterrows():
#     print(row['cm1'], row['cm2'])

# FUNCTIONS FOR CALCULATING NASH EQUILIBRIA.
# Define a function to calculate the expected payoff for a given player and strategy profile
def expected_payoff(payoff_matrix, player, strategy_profile):
    
    opponent = 1 - player
    
    sum_ = 0
    a = payoff_matrix[player]
    for i in range(3):
        for j in range(2):
            sum_ = sum_ + a[i][j]
    b = strategy_profile[player]
    c = strategy_profile[opponent]
    return sum_ + b + c

# Define a function to find the best response for a given player and opponent strategy
def best_response(row, payoff_matrix, player, opponent_strategy):
    best_payoff = float('-inf')
    best_strategy = None
    for i in range(3):
        strategy_profile = (i, opponent_strategy)
        sum_ = row['g'] + row['cm1'] + row['cm2'] + row['ci1'] + row['ci2'] + row['l'] + row['a'] + row['b']
        payoff = random.randint(1, int(sum_)) + expected_payoff(payoff_matrix, player, strategy_profile)
        # payoff = random.randint(1, 100) + expected_payoff(payoff_matrix, player, strategy_profile)
        if payoff > best_payoff:
            best_payoff = payoff
            best_strategy = i
    return best_strategy

# DISPLAY THE PAYOFF MATRIX.
def displayMatrix(mat):
    print("PAYOFF MATRIX:")
    for i in range(3):
        for j in range(3):
            print(mat[i][j], end=" ")
        print()
    print()

# CALCULATE THE PAYOFF MATRIX.
class CalculatePayoffMatrix:

    def defenseAttack00():
        return [0, 0]

    def defenseAttack01(ci1):
        return [0, round(-ci1,2)]

    def defenseAttack02(ci2):
        return [0, round(ci2, 2)]

    def defenseAttack10(cm1, g):
        return [round(cm1 + g, 2), -1]

    def defenseAttack11(cm1, a, g, ci1, l):
        return [round(-cm1 + (1 - a) * g, 2), round(-ci1 - (1 - a) * l, 2)]

    def defenseAttack12(cm1, b, g, ci2, l):
        return [round(-cm1 + (1 - b)*g, 2), round(-ci2-(1-b)*l, 2)]

    def defenseAttack20(cm2, g):
        return [round(-cm2 + g, 2), -1]

    def defenseAttack21(cm2, a, g, ci1, l):
        return [round(-cm2 + (1-a)*g, 2), round(-ci1 - (1 - a) * l, 2)]

    def defenseAttack22(cm2, b, g, ci1, l):
        return [round(-cm2 + (1 - b) * g, 2), round(-ci1 - (1 - b) * l, 2)]


for index, row in df.iterrows():

    mat = [[0 for _ in range(3)] for _ in range(3)]

    obj = CalculatePayoffMatrix

    for i in range(3):
        for j in range(3):
            if i==0 and j==0:
                mat[i][j] = obj.defenseAttack00()
            if i==0 and j==1:
                mat[i][j] = obj.defenseAttack01(row['ci1'])
            if i==0 and j==2:
                mat[i][j] = obj.defenseAttack02(row['ci2'])
            if i==1 and j==0:
                mat[i][j] = obj.defenseAttack10(row['cm1'], row['g'])
            if i==1 and j==1:
                mat[i][j] = obj.defenseAttack11(row['cm1'], row['a'], row['g'], row['ci1'], row['l'])
            if i==1 and j==2:
                mat[i][j] = obj.defenseAttack12(row['cm1'], row['b'], row['g'], row['ci2'], row['l'])
            if i==2 and j==0:
                mat[i][j] = obj.defenseAttack20(row['cm2'], row['g'])
            if i==2 and j==1:
                mat[i][j] = obj.defenseAttack21(row['cm2'], row['a'], row['g'], row['ci1'], row['l'])
            if i==2 and j==2:
                mat[i][j] = obj.defenseAttack22(row['cm2'], row['b'], row['g'], row['ci1'], row['l'])

    # displayMatrix(mat)
    payoff_matrix = mat

    # Define the payoff matrix
    # payoff_matrix = [[[1, 2], [3, 4], [5, 6]],
    #                 [[7, 8], [9, 10], [11, 12]],
    #                 [[13, 14], [15, 16], [17, 18]]]

    # Find the Nash equilibria
    nash_equilibria = []
    for i in range(3):
        for j in range(3):
            strategy_profile = (i, j)
            if (best_response(row, payoff_matrix, 0, j) == i) and (best_response(row, payoff_matrix, 1, i) == j):
                nash_equilibria.append(strategy_profile)

    # Print the Nash equilibria
    print("The Nash equilibria are: ", end=" ")
    if(not len(nash_equilibria)):
        print("No nash equilibria.", end="")
    else:
        print("[ " + str(len(nash_equilibria)) + " ] -> ", end=" ")
    for eq in nash_equilibria:
        print(eq, end=" ")
    print()