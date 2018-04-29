import numpy as np
import pandas as pd

def feature_ranking(X, X_train, clf):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print('\nFeature ranking:')
    features_names = X.columns.values
    for f in range(X_train.shape[1]):
        print('{0}# {1}: {2}%'.format(f + 1, features_names[indices[f]], round(importances[indices[f]] * 100, 2)))

def model_accuracy(clf, X_train, Y_train, X_test, Y_test):
    print('\nTraining accuracy: {0}%'.format(round(clf.score(X_train, Y_train) * 100, 2)))
    print('Testing accuracy: {0}%'.format(round(clf.score(X_test, Y_test) * 100, 2)))

def output(data_test, clf):
    ids = data_test['PassengerId']
    predictions = clf.predict(data_test.drop('PassengerId', axis=1))
    output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
    output.to_csv('estimation.csv', index=False)
    print("\nOutput written to estimation.csv")
