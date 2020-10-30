#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

from model import SvmSmo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd


def main():
    df = pd.read_csv('SMO_Assignment_Dataset.txt', sep='\s+', header=None)
    df[0] = df[0].astype(float)
    df[1] = df[1].astype(float)
    df[2] = df[2].astype(int)
    X = df.values[:, :2]
    y = df.values[:, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.fit_transform(X_test, y_test)

    model = SvmSmo()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    model.visualize(save=False)


if __name__ == '__main__':
    main()
