import pandas as pd
from sklearn.model_selection import train_test_split

def transform_features(data_train, data_test, selected_features):
    df_train = data_train[selected_features + ['Survived']].copy()
    df_test = data_test[selected_features].copy()
    age_mean = pd.concat([data_train['Age'], data_test['Age']], ignore_index=True).mean()
    fare_mean = pd.concat([data_train['Fare'], data_test['Fare']], ignore_index=True).mean()

    df_train['Sex'] = df_train['Sex'].map({'male': 1, 'female': 2}).astype(int)
    df_train['Age'] = df_train['Age'].fillna(age_mean)
    df_train['Fare'] = df_train['Fare'].fillna(fare_mean)
    df_train = df_train.dropna()
    df_train['Embarked'] = df_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    df_test['Sex'] = df_test['Sex'].map({'male': 1, 'female': 2}).astype(int)
    df_test['Age'] = df_test['Age'].fillna(age_mean)
    df_test['Fare'] = df_test['Fare'].fillna(fare_mean)
    df_test = df_test.dropna()
    df_test['Embarked'] = df_test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    return df_train, df_test

def features_scaling(df_train, df_test, selected_features):
    for feature in selected_features:
        mean = pd.concat([df_train[feature], df_test[feature]], ignore_index=True).mean()
        std = pd.concat([df_train[feature], df_test[feature]], ignore_index=True).std()
        df_train[feature] = (df_train[feature] - mean) / std
        df_test[feature] = (df_test[feature] - mean) / std
    return df_train, df_test

def split_data(df_train, df_test, selected_features):
    X = df_train.drop(['Survived'], axis=1)
    Y = df_train['Survived']
    # Split training data to 80/20
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    Y_train = Y_train.values.reshape(-1, 1)
    Y_test = Y_test.values.reshape(-1, 1)
    test_X = df_test[selected_features].copy()
    return X_train, Y_train, X_test, Y_test, test_X
