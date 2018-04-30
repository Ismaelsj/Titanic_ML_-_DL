import os.path
from sys import argv
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import data_process
import model_process
import estimations_and_accuracy

def main():
        # Get the data
    data_train = pd.read_csv('dataset/train.csv')
    data_test = pd.read_csv('dataset/test.csv')
        # Transforming and dividing features
    data_train, data_test = data_process.transform_features(data_train, data_test)
    X, Y, X_train, X_test, Y_train, Y_test = data_process.training_features(data_train)

        # Build/Get model
    if (len(argv) > 1 and argv[1] == '-n') and (os.path.isfile('.clf_model.pkl') == True):
        clf = joblib.load('.clf_model.pkl')
    else:
        if (len(argv) > 1 and argv[1] == '-n'):
            print("\nNo model found.\nBuilding a new one...")
        clf = model_process.build_clf_model(X_train, Y_train)

      # Print features importance
    estimations_and_accuracy.feature_ranking(X, X_train, clf)
        # Print accuracy
    estimations_and_accuracy.model_accuracy(clf, X_train, Y_train, X_test, Y_test)
        # Output the submission to estimation.csv
    estimations_and_accuracy.output(data_test, clf)

if __name__ == '__main__':
    main()
