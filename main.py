import numpy as np
import pandas as pd
import lib


def main():
    df = pd.read_csv('./data/student-mat.csv')
    df.G3[df.G3 <= 9] = 1
    df.G3[(9 < df.G3) & (df.G3 <= 11)] = 2
    df.G3[(11 < df.G3) & (df.G3 <= 13)] = 3
    df.G3[(13 < df.G3) & (df.G3 <= 15)] = 4
    df.G3[df.G3 > 15] = 5
    df.drop(columns=['G1', 'G2', 'absences', 'higher', 'school'], inplace=True)
    df = df.astype(str)
    for col in df.columns:
        df[col] = col + '_' + df[col]

    L, suppData = lib.apriori(df.to_numpy(), minSupport=0.2)
    rules = lib.generateRules(L, suppData, minConf=0.9)
    rules = sorted(rules, key=lambda x: x[2], reverse=True)
    for i in rules:
        print(i)
    # print(L)


if __name__ == '__main__':
    main()
