
import numpy as np
import pandas as pd


class Kaggle_DT:
    def read_file(self, features, labels):
        X = pd.read_csv(features, header=None,)
        Y = pd.read_csv(labels, header=None,)

        X = X.drop(X.columns[[0]], axis=1)
        Y = Y.iloc[1:]
        print(X.head())
        print(Y.head())
        return X, Y


def main():
    dt_classifier = Kaggle_DT()
    dt_classifier.read_file("../Data/train_features.csv", "../Data/train_label.csv")
    print("HEL")

if __name__ == '__main__':
    main()