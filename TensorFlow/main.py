import os.path
from sys import argv
import numpy as np
import pandas as pd
import data_process
import model
import accuracy_estimation

def main():
        # Get the data
    data_train = pd.read_csv('dataset/train.csv')
    data_test = pd.read_csv('dataset/test.csv')
        # Transforming and dividing features
    Id_test = data_test['PassengerId']
    selected_features = ['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df_train, df_test = data_process.transform_features(data_train, data_test, selected_features)
    df_train, df_test = data_process.features_scaling(df_train, df_test, selected_features)
    X_train, Y_train, X_test, Y_test, test_X = data_process.split_data(df_train, df_test, selected_features)
        # Set parameters
    parameters = {}
    parameters['model_path'] = 'model/Titanic.ckpt'
    parameters['n_input'], parameters['n_features'] = X_train.shape
    parameters['n_hidden'] = 2
    parameters['hidden_dim'] = 40
    parameters['n_class'] = 1
    parameters['learning_rate'] = 0.01
    parameters['training_epochs'] = 15000
    parameters['visualize'] = False
    if ((len(argv) > 1 and argv[1] == '-v') or (len(argv) > 2 and argv[2] == '-v')):
        parameters['visualize'] = True

        # Get model & train
    titanic_model = model.make_model(parameters)
    if (len(argv) > 1 and argv[1] == '-n') or (len(argv) > 2 and argv[2] == '-n'):
        model.neural_network(X_train, Y_train, parameters, titanic_model, X_test, Y_test)
        # Print accuracy
    if os.path.isfile(parameters['model_path']) == True:
        accuracy_estimation.Accuracy(parameters, titanic_model, X_train, Y_train, X_test, Y_test)
        # Output the submission to estimation.csv
    if os.path.isfile(parameters['model_path']) == True:
        accuracy_estimation.Estimation(parameters, titanic_model, test_X, Id_test)
    else:
        print("\nNo model found, please create a new file named 'Titanic.ckpt' in a directory named 'model' and launch the programme with th folowing commande :\n'python3 main.py -n'\n")

if __name__ == '__main__':
    main()
