import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

def get_clf_parameters(clf, X_train, y_train):
    parameters = {'n_estimators': [4, 6, 9],
                    'max_features': ['log2', 'sqrt','auto'],
                    'criterion': ['entropy', 'gini'],
                    'max_depth': [2, 3, 5, 10],
                    'random_state': [42],
                    'min_samples_split': [2, 3, 5],
                    'min_samples_leaf': [1,5,8]}
        # Run grid search to get the best parameters
    grid = GridSearchCV(clf, parameters)
    grid = grid.fit(X_train, y_train)
    print("\nSearching for best parameters:")
    print("Best score : {0}".format(grid.best_score_))
    print("Best parameters : {0}".format(grid.best_params_))
        # Set the best parameters to clf
    clf = grid.best_estimator_
        # Training model
    clf.fit(X_train, y_train)
    return clf

def build_clf_model(X_train, y_train):
    clf = RandomForestClassifier()
    clf = get_clf_parameters(clf, X_train, y_train)
        # Output submission to estimation.csv
    joblib.dump(clf, '.clf_model.pkl')
    return clf
