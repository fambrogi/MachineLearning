import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from clean_analyze_data import load_clean_data, data


def regTree_f(ds, train_df, test_df, target):

    x_train = train_df.drop(columns = [target])
    y_train = train_df[target]
    x_test = test_df.drop(columns = [target])
    y_test = test_df[target]

    def rss(y_left, y_right):
        def squared_residual_sum(y):
            return np.sum((y - np.mean(y)) ** 2)
        return squared_residual_sum(y_left) + squared_residual_sum(y_right)

    def compute_rss_by_threshold(feature):
        features_rss = []
        thresholds = x_train[feature].unique().tolist()
        thresholds.sort()
        thresholds = thresholds[1:]
        for t in thresholds:
            y_left_ix = X_train[feature] < t
            y_left, y_right = y_train[y_left_ix], y_train[~y_left_ix]
            features_rss.append(rss(y_left, y_right))


        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.xlabel('Thresholds')
        plt.ylabel(feature)
        plt.title(feature + ' RSS')
        plt.plot(thresholds, rss)

        plt.tight_layout()
        plt.savefig('Plots/results/' + ds + '/' + ds + '_rss.png', dpi=150)

        dummy = plot(ds, feature)
        return thresholds, features_rss


    def find_best_rule(X_train, y_train):
        best_feature, best_threshold, min_rss = None, None, np.inf
        for feature in X_train.columns:
            thresholds = X_train[feature].unique().tolist()
            thresholds.sort()
            thresholds = thresholds[1:]
            for t in thresholds:
                y_left_ix = X_train[feature] < t
                y_left, y_right = y_train[y_left_ix], y_train[~y_left_ix]
                t_rss = rss(y_left, y_right)
                if t_rss < min_rss:
                    min_rss = t_rss
                    best_threshold = t
                    best_feature = feature

        return {'feature': best_feature, 'threshold': best_threshold}



    def split(x_train, y_train, depth, max_depth):
        if depth == max_depth or len(x_train) < 2:
            return {'prediction': np.mean(y_train)}

        rule = find_best_rule(x_train, y_train)
        left_ix = x_train[rule['feature']] < rule['threshold']
        rule['left'] = split(x_train[left_ix], y_train[left_ix], depth + 1, max_depth)
        rule['right'] = split(x_train[~left_ix], y_train[~left_ix], depth + 1, max_depth)
        return rule

    rules = split(X_train, y_train, 0, 3)


    def predict(sample, rules):
        prediction = None
        while prediction is None:
            feature, threshold = rules['feature'], rules['threshold']
            if sample[feature] < threshold:
                rules = rules['left']
            else:
                rules = rules['right']
            prediction = rules.get('prediction', None)
        return prediction

    def evaluate(X, y):
        preds = X.apply(predict, axis='columns', rules=rules.copy())
        return r2_score(preds, y)


    X_train, y_train, X_test, y_test = prepare_dataset()

    for max_depth in range(3, 11):
        rules = split(x_train, y_train, 0, max_depth)
        train_r2 = evaluate(x_train, y_train)
        test_r2 = evaluate(x_test, y_test)
        print('Max Depth', max_depth, 'Training R2:', train_r2, 'Test R2:',test_r2)




def treeF(ds, train_df, test_df, target):

    x_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    x_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    def find_best_rule(x_train, y_train):
        best_feature, best_threshold, min_rss = None, None, np.inf
        for feature in x_train.columns:
            thresholds = x_train[feature].unique().tolist()
            thresholds.sort()
            thresholds = thresholds[1:]
            for t in thresholds:
                y_left_ix = x_train[feature] < t
                y_left, y_right = y_train[y_left_ix], y_train[~y_left_ix]
                t_rss = rss(y_left, y_right)
                if t_rss < min_rss:
                    min_rss = t_rss
                    best_threshold = t
                    best_feature = feature

        return {'feature': best_feature, 'threshold': best_threshold}


    def split(x_train, y_train, depth, max_depth):

        if depth == max_depth or len(X_train) < 2:
            return {'prediction': np.mean(y_train)}

        rule = find_best_rule(X_train, y_train)
        left_ix = x_train[rule['feature']] < rule['threshold']
        rule['left'] = split(x_train[left_ix], y_train[left_ix], depth + 1, max_depth)
        rule['right'] = split(x_train[~left_ix], y_train[~left_ix], depth + 1, max_depth)
        return rule


    rules = split(X_train, y_train, 0, 3)

    def predict(sample, rules):
        prediction = None
        while prediction is None:
            feature, threshold = rules['feature'], rules['threshold']
            if sample[feature] < threshold:
                rules = rules['left']
            else:
                rules = rules['right']
            prediction = rules.get('prediction', None)
        return prediction

    predict()










