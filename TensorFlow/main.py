import os.path
from sys import argv
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import data_process
import model
import accuracy

def main():
        # Get the data
    data_train = pd.read_csv('dataset/train.csv')
    data_test = pd.read_csv('dataset/test.csv')
        # Transforming and dividing features
    data_train, data_test = data_process.transform_features(data_train, data_test)
    X, Y, X_train, X_test, Y_train, Y_test = data_process.training_features(data_train)

        # Set parameters
    parameters = {}
    parameters['model_name'] = 'Titanic.ckpt'
    parameters['n_input'], parameters['n_features'] = X_train.shape
    parameters['n_hidden'] = 2
    parameters['hidden_dim'] = 10
    parameters['n_class'] = 1
    parameters['learning_rate'] = 0.03
    parameters['training_epochs'] = 3000
    parameters['batch_size'] = 20
    parameters['visualize'] = False
    if (len(argv) > 1 and argv[1] == '-v'):
        parameters['visualize'] = True

        # Get model & train
    titanic_model = model.make_model(parameters)
    model.neural_network(X_train, Y_train, parameters, titanic_model)
        # Print accuracy
    #accuracy.Accuracy(parameters, model, X_train, Y_train, X_test, Y_test)
        # Output the submission to estimation.csv
    # ...

if __name__ == '__main__':
    main()
