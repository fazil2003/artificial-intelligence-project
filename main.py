import pandas as pd

df = pd.read_csv("finaldataset.csv")

# for index, row in df.iterrows():
#     print(row['cm1'], row['cm2'])

def defenseAttack00():
    return (0, 0)

def defenseAttack01(ci1):
    return (0, round(-ci1,2))

def defenseAttack02(ci2):
    return (0, round(ci2, 2))

def defenseAttack10(cm1, g):
    return (round(cm1 + g, 2), -1)

def defenseAttack11(cm1, a, g, ci1, l):
    return (round(-cm1 + (1 - a) * g, 2), round(-ci1 - (1 - a) * l, 2))

def defenseAttack12(cm1, b, g, ci2, l):
    return (round(-cm1 + (1 - b)*g, 2), round(-ci2-(1-b)*l, 2))

def defenseAttack20(cm2, g):
    return (round(-cm2 + g, 2), -1)

def defenseAttack21(cm2, a, g, ci1, l):
    return (round(-cm2 + (1-a)*g, 2), round(-ci1 - (1 - a) * l, 2))

def defenseAttack22(cm2, b, g, ci1, l):
    return (round(-cm2 + (1 - b) * g, 2), round(-ci1 - (1 - b) * l, 2))


for index, row in df.iterrows():

    print("PAYOFF MATRIX:")

    mat = [[0 for _ in range(3)] for _ in range(3)]

    for i in range(3):
        for j in range(3):
            if i==0 and j==0:
                mat[i][j] = defenseAttack00()
            if i==0 and j==1:
                mat[i][j] = defenseAttack01(row['ci1'])
            if i==0 and j==2:
                mat[i][j] = defenseAttack02(row['ci2'])
            if i==1 and j==0:
                mat[i][j] = defenseAttack10(row['cm1'], row['g'])
            if i==1 and j==1:
                mat[i][j] = defenseAttack11(row['cm1'], row['a'], row['g'], row['ci1'], row['l'])
            if i==1 and j==2:
                mat[i][j] = defenseAttack12(row['cm1'], row['b'], row['g'], row['ci2'], row['l'])
            if i==2 and j==0:
                mat[i][j] = defenseAttack20(row['cm2'], row['g'])
            if i==2 and j==1:
                mat[i][j] = defenseAttack21(row['cm2'], row['a'], row['g'], row['ci1'], row['l'])
            if i==2 and j==2:
                mat[i][j] = defenseAttack22(row['cm2'], row['b'], row['g'], row['ci1'], row['l'])

    for i in range(3):
        for j in range(3):
            print(mat[i][j], end=" ")
        print()

    print()