
# coding: utf-8

# In[ ]:




from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
#deep
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

import datetime
import pandas as pd
import numpy as np
import time


def log(text, t_start=None):
    if t_start is None:
        print(text)
    else:
        elapsed_time = round(time.time() - t_start, 2)
        print(text + "\t(" + str(elapsed_time) + "s)")

t = time.time()
customers = pd.read_csv("data/customers.csv")
products = pd.read_csv("data/products.csv")
x_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
x_test = pd.read_csv("data/X_test.csv")
log("files loaded", t)

# SizeAdviceDescription
SizeAdviceDescriptionCleaner = {}
SizeAdviceDescriptionCleaner['nan'] = 0
SizeAdviceDescriptionCleaner['Ce mod\xc3\x83\xc2\xa8le chausse normalement'] = 0
SizeAdviceDescriptionCleaner['Mod\xc3\x83\xc2\xa8le confortable, convient aux pieds larges'] = -.5
SizeAdviceDescriptionCleaner['Mod\xc3\x83\xc2\xa8le \xc3\x83\xc2\xa9troit, convient aux pieds fins'] = .5
SizeAdviceDescriptionCleaner['Prenez votre pointure habituelle'] = 0
SizeAdviceDescriptionCleaner['Chaussant particuli\xc3\x83\xc2\xa8rement g\xc3\x83\xc2\xa9n\xc3\x83\xc2\xa9reux. Nous vous conseillons de choisir deux tailles en dessous de votre pointure habituelle.'] = -2
SizeAdviceDescriptionCleaner['Chaussant petit. Si vous \xc3\x83\xc2\xaates habituellement entre deux pointures, nous vous conseillons de choisir une demi taille au-dessus de votre pointure habituelle.'] = .5
SizeAdviceDescriptionCleaner['Prenez une taille au-dessus de sa pointure !'] = 1
SizeAdviceDescriptionCleaner['Prenez une taille au-dessus de votre pointure habituelle'] = 1
SizeAdviceDescriptionCleaner['Prenez une taille en dessous de sa pointure !'] = -1
SizeAdviceDescriptionCleaner['Prenez une taille en dessous de votre pointure habituelle'] = -1

# BirthDate
def age(birthdate):
    if type(birthdate) == type(" "):
        return 2016 - int(birthdate[:4])
    return None

# OrderCreationDate and SeasonLabel
def order_season(orderdate):
    month = int(orderdate[5:7])
    if month >= 4 and month <= 9:
        return "Printemps/Et\xc3\x83\xc2\xa9"
    return "Automne/Hiver"


def build_df(x):
    """Builds a pandas DataFrame with clean columns from a read CSV"""
    
    t = time.time()
    m = None
    
    # join
    m = pd.merge(x, products, how='left', on='VariantId', suffixes=('_pr', ''))
    m = pd.merge(m, customers, how='left', on='CustomerId', suffixes=('_cs', ''))
    
    # converting UnitPMPEUR
    m.UnitPMPEUR = m["UnitPMPEUR"].map(lambda row: float(row.replace(',', '.')))
    
    # building news columns
    m["MatchGender"] = m["Gender"] == m["GenderLabel"]
    m["MatchSeason"] = m["SeasonLabel_pr"] == m["SeasonLabel"]
    m["OrderSeason"] = m["OrderCreationDate"].map(order_season)
    m["MatchOrderSeason"] = m["OrderSeason"] == m["SeasonLabel"]
    
    # cleaning
    m["SizeAdviceDescription"] = m["SizeAdviceDescription"].map(SizeAdviceDescriptionCleaner)
    m["BirthDate"] = m["BirthDate"].map(age)
        
    # removing useless columns
    blacklist = ['VariantId', 'CustomerId', 'OrderNumber', 'LineItem',
                 'ProductColorId', 'BrandId', 'SupplierColor', 'OrderShipDate',
                 'ProductId', 'BillingPostalCode', 'FirstOrderDate',
                 'OrderStatusLabel', 'MinSize', 'MaxSize', 'OrderSeason',
                 'OrderCreationDate', 'SubtypeLabel', 'ProductType'
                ]
    whitelist = None
    if blacklist is not None:
        m = m.drop(blacklist, axis=1)
    if whitelist is not None:
        for col in m.columns:
            if col not in whitelist:
                m = m.drop([col], axis=1)

    print "dataframe shape:", m.shape
    log("dataframe built", t)
    return m


df_test = build_df(x_test)
df_train = build_df(x_train)


def mask(m):
    columns2bin = [col for col in m.columns if m[col].dtype == 'object']
    other_cols = m.drop(columns2bin, axis=1)
    new_cols = pd.get_dummies(m.loc[:, columns2bin])
    res = pd.concat([other_cols, new_cols], axis=1)
    res = res.fillna(0)
    print "new shape:", res.shape
    return res

def compute(name, clf, x1, x2, slc=100000):
    print "\n-----", name, "-----"
    clf.fit(x1.iloc[:slc], y_train.ReturnQuantityBin[:slc])
    
    predict_train = clf.predict_proba(x1.iloc[:slc])
    score_train = roc_auc_score(y_train.ReturnQuantityBin[:slc], predict_train[:, 1])
    print "train score:", score_train
    
    predict_test = clf.predict_proba(x1.iloc[slc:2 * slc])
    score_test = roc_auc_score(y_train.ReturnQuantityBin[slc:2 * slc], predict_test[:, 1])
    print "test score:", score_test
    return score_train, score_test

def compute_all(x1, x2, slc=100000):
    """Tries different classifiers and returns the best one (best test score)"""
    t = time.time()
    best_index, best_score = None, None
    
    print "train shape:\t", x1.shape, "\t", y_train.shape
    print "test shape:\t", x2.shape, "\t", y_test.shape
    
    classifiers = [("random forest", RandomForestClassifier()),
                   ("decision tree", DecisionTreeClassifier()),
                   ("logistic regression", LogisticRegression()),
                    ("DEEP",MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(100, 10), random_state=1))]
    
    for i, (name, clf) in enumerate(classifiers):
        score_train, score_test = compute(name, clf, x1, x2, slc)
        if best_score is None or score_test > best_score:
            best_index, best_score = i, score_test
    
    log("\nbest classifier: " + classifiers[best_index][0], t)
    return classifiers[best_index][1]

def output(clf, x1, x2):
    t = time.time()
    y_tosubmit = clf.predict_proba(x2.loc[:, x1.columns].fillna(0))
    
    timestamp = '{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
    filename = "ypred_{0}_sgd.txt".format(timestamp)
    np.savetxt(filename, y_tosubmit[:,1], fmt='%f')
    
    f = open("predictions.txt", 'a')
    f.write(timestamp + '\n' + repr(clf).replace('\n          ', '') + '\n\n')
    f.close()
    
    print "shape:", y_tosubmit.shape
    log("generated output at " + filename, t)


t = time.time()
x1 = mask(df_train)
x2 = mask(df_test)
log("applied mask", t)



def shuffle(x, y, steps=10, slc=100000, plot=True):
    scores_train, scores_test = [], []
    best_clf, best_score = None, None
    
    z = x.copy(deep=True)
    z["ReturnQuantityBin"] = y.ReturnQuantityBin
    
    for k in range(steps):
        u = z.sample(frac=1)
        v = u.loc[:, ["ReturnQuantityBin"]]
        u = u.drop(["ReturnQuantityBin"], axis=1)
        
        #clf = LogisticRegression()
        clf = MLPRegressor(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=(100, 100,100,100), random_state=1)
        clf.fit(u.iloc[:slc], v.ReturnQuantityBin[:slc])
        
        predict_train = clf.predict_proba(u.iloc[:slc])
        score_train = roc_auc_score(v.ReturnQuantityBin[:slc], predict_train[:, 1])
    
        predict_test = clf.predict_proba(u.iloc[slc:2 * slc])
        score_test = roc_auc_score(v.ReturnQuantityBin[slc:2 * slc], predict_test[:, 1])
        
        if best_clf is None or score_test > best_score:
            best_clf, best_score = clf, score_test
        
        if plot:
            print "test", k, "\ttrain:", score_train, "\ttest:", score_test
        
        scores_train.append(score_train)
        scores_test.append(score_test)
    
    if plot:
        plt.figure(figsize=(16, 10))
        plt.xlabel("train score")
        plt.ylabel("test score")
        plt.plot(scores_train, scores_test, '+')
        plt.show()
    
    return scores_train, scores_test, best_clf, best_score


sc_train, sc_test, clf, score = shuffle(x1, y_train, slc=100000, steps=10, plot=False)



output(clf, x1, x2)

