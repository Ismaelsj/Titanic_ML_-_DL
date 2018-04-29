import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df

def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(data_train, data_test):
    data_train = simplify_ages(data_train)
    data_test = simplify_ages(data_test)
    data_train = simplify_cabins(data_train)
    data_test = simplify_cabins(data_test)
    data_train = simplify_fares(data_train)
    data_test = simplify_fares(data_test)
    data_train = format_name(data_train)
    data_test = format_name(data_test)
    data_train = drop_features(data_train)
    data_test = drop_features(data_test)
        # Encoding features
    data_train, data_test = encode_features(data_train, data_test)
    return data_train, data_test

def training_features(data_train):
    X = data_train.drop(['Survived', 'PassengerId'], axis=1)
    Y = data_train['Survived']
        # Split the data to 80/20
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X, Y, X_train, X_test, Y_train, Y_test
