# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
# import random
# import sys
# import os


def funk_mask(d):
    """ Defining a simple mask over the input data """
    columns_ext = ["OrderCreationDate", "OrderNumber", "VariantId",
                   "CustomerId", "OrderCreationDate", "OrderShipDate",
                   "BillingPostalCode"]

    # Remove columns from input array
    x1 = d.loc[:, [xx for xx in d.columns if xx not in columns_ext]]

    # Convert UnitPMPEur column to floats (price of a unit, in euros)
    x1.UnitPMPEUR = [np.float(x.replace(",", ".")) for x in x1.UnitPMPEUR]

    # Select columns that contains string values
    columns2bin = [x for x in x1.columns if x1[x].dtype == np.dtype('O')]
    # Convert those to numerical indicators
    x2 = pd.get_dummies(x1.loc[:, columns2bin])
    # Extract other columns without modyfing them
    x3 = x1.loc[:, [xx for xx in x1.columns if xx not in columns2bin]]

    # Rebuild data
    res = pd.concat([x3, x2], axis=1)
    # Fill holes with 0s
    res = res.fillna(0)
    return res


def log(text, t_start=None):
    if t_start is None:
        print(text)
    else:
        elapsed_time = round(time.time() - t_start, 2)
        print(text + "\t(" + str(elapsed_time) + "s)")


def main():
    log("enter main")

    t = time.time()
    # customers = pd.read_csv("../data/customers.csv")
    # products = pd.read_csv("../data/products.csv")
    x_train = pd.read_csv("../data/X_train.csv")
    y_train = pd.read_csv("../data/y_train.csv")
    x_test = pd.read_csv("../data/X_test.csv")
    y_test = pd.read_csv("../data/y_test.csv")
    log("files loaded", t)

    t = time.time()
    x1 = funk_mask(x_train)
    x2 = funk_mask(x_test)
    log("applied mask", t)

    t = time.time()
    clf = LogisticRegression()
    clf.fit(x1.iloc[:50000], y_train.ReturnQuantityBin[:50000])
    log("fit classifier", t)

    yres = clf.predict_proba(x1.loc[:100000, x1.columns])
    print("train score:", roc_auc_score(y_train.ReturnQuantityBin.iloc[:100001],
                                        yres[:, 1]))

    y_tosubmit = clf.predict_proba(x2.loc[:100000, x1.columns])
    print("test score:", roc_auc_score(y_test.ReturnQuantityBin.iloc[:100001],
                                       y_tosubmit[:, 1]))


def check_value_error(m):
    columns = [c for c in m.columns if m[c].dtype == 'float64']
    print(m.loc[:, columns].dtypes)
    print("Any NaN?\t", np.any(np.isnan(m)))
    print("All finite?\t", np.all(np.isfinite(m)))


if __name__ == '__main__':
    main()
