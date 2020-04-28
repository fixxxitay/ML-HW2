import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from exploreData import explore_data
from features import nominal_features, integer_features, float_features
import featureSelection
import exploreData

def train_test_validation_split(x, test_size, validation_size):
    """
    :param x: features
    :param y: labels
    :param test_size: test set size in percentage (0 < test_size < 1)
    :param validation_size: validation set size in percentage (0 < validation_size < 1)
    :return X_train, X_test, y_train, y_test
    """
    n_samples = x.shape[0]
    num_train = int((1 - test_size - validation_size) * n_samples)
    num_test = int(test_size * n_samples)
    rand_gen = np.random.RandomState()

    x_train = []
    x_test = []
    x_validation = []

    indexes = rand_gen.permutation(n_samples)

    for i in range(n_samples):
        if i < num_train:
            x_train.append(x.iloc[indexes[i]])
        elif (i >= num_train) & (i < num_train + num_test):
            x_test.append(x.iloc[indexes[i]])
        else:
            x_validation.append(x.iloc[indexes[i]])
    return pd.DataFrame.from_records(x_train), pd.DataFrame.from_records(x_test), pd.DataFrame.from_records(x_validation)

def deterministicSplit(df, train, test):
    df_train = df.iloc[0:round(len(df) * train), :]
    df_test = df.iloc[round(len(df) * train):round(len(df) * (train+test)), :]
    df_validation = df.iloc[round(len(df) * (train+test)):len(df), :]

    return df_train, df_test, df_validation

def save_files(df_train, df_test, df_validation):
    df_train.to_csv('train.csv', index=False)
    df_test.to_csv('test.csv', index=False)
    df_validation.to_csv('validation.csv', index=False)

def get_filter_selection(df_train):
    # The filter method : correlation factor between features
    # Remove the highly correlated ones
    correlated_features = set()
    correlation_matrix = df_train.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
  
    ret = np.ones(df_train.shape[1]-1)
    for i in range(len(ret)):
        if df_train.columns[i+1] in correlated_features:
            ret[i] = 0

    return ret

def get_wrapper_selection(df_train):
    # Wrapper method :
    model = LogisticRegression()
    rfe = RFE(model, 16)
    fit = rfe.fit(df_train.values[:, 1:], df_train.values[:, 0])
    #print("Num Features: %s" % (fit.n_features_))
    #print("Selected Features: %s" % (fit.support_))
    #print("Feature Ranking: %s" % (fit.ranking_))

    return fit.support_

def remove_wrong_party_and_na(df_train, df_test, df_validation):
    df_train = df_train[df_train.Vote != 10]
    df_train = df_train[df_train.Vote != 4]
    df_train = df_train.dropna()

    df_test = df_test[df_test.Vote != 10]
    df_test = df_test[df_test.Vote != 4]
    df_test = df_test.dropna()

    df_validation = df_validation[df_validation.Vote != 10]
    df_validation = df_validation[df_validation.Vote != 4]
    df_validation = df_validation.dropna()
    
    return df_train, df_test, df_validation

def save_raw_data(df_test, df_train, df_validation):
    df_train.to_csv('raw_train.csv', index=False)
    df_test.to_csv('raw_test.csv', index=False)
    df_validation.to_csv('raw_validation.csv', index=False)

def complete_missing_values(df_train: pd.DataFrame, df_test: pd.DataFrame, df_validation: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df_train = df_train[df_train > 0]
    df_test = df_test[df_test > 0]
    df_validation = df_validation[df_validation > 0]

    for col in df_train.columns.values:
        if col == 'Vote':
            df_train[col].fillna(df_train[col].mode()[0], inplace=True)
            continue

        filler = None
        if col in nominal_features:
            filler = df_train[col].mode()[0]

        if col in integer_features:
            filler = round(df_train[col].mean())

        if col in float_features:
            filler = df_train[col].mean()

        df_train[col].fillna(filler, inplace=True)
        df_test[col].fillna(filler, inplace=True)
        df_validation[col].fillna(filler, inplace=True)
    
    return df_train, df_test, df_validation

def nominal_to_numerical_categories(df):
    # from nominal to Categorical
    df = df.apply(lambda x: pd.Categorical(x) if x.dtype != 'float64' else x, axis=0)
    # give number to each Categorical
    df = df.apply(lambda x: x.cat.codes if x.dtype != 'float64' else x, axis=0)
    return df

def apply_feature_selection(df_train, df_test, df_validation, featureSet):
    arrayBool = np.array([True], dtype=bool)
    arrayBool = np.append(arrayBool, featureSet)
    df_train = df_train.iloc[:, arrayBool==True]
    df_test = df_test.iloc[:, arrayBool==True]
    df_validation = df_validation.iloc[:, arrayBool==True]
    
    return df_train, df_test, df_validation


def main():
    df = pd.read_csv("ElectionsData.csv")

    # Convert nominal types to numerical categories
    df = nominal_to_numerical_categories(df)

    # split the data to train , test and validation
    df_train, df_test, df_validation = train_test_validation_split(df, 0.2, 0.2)

    # Save the raw data first
    #save_raw_data(df_test, df_train, df_validation)
    
    # 1 - Imputation - Complete missing values
    df_train, df_test, df_validation = complete_missing_values(df_train, df_test, df_validation)

    # Remove lines with wrong party (Violets | Khakis)
    df_train, df_test, df_validation = remove_wrong_party_and_na(df_train, df_test, df_validation)

    # 2 - Data Cleansing
    # Outlier detection using z score

    # 3 - Normalization (scaling)

    # print some graph about the data
    #explore_data(df)

    # 4 - Feature Selection
    featureSet = get_filter_selection(df_train)
    df_train, df_test, df_validation = apply_feature_selection(df_train, df_test, df_validation, featureSet)

    #featureSet = get_wrapper_selection(df_train)
    #df_train, df_test, df_validation = apply_feature_selection(df_train, df_test, df_validation, featureSet)
    #print("Score for Regression: ")
    #print(featureSelection.getScore(df_test.iloc[:, 1:], df_test.iloc[:, 0]))

    #featureSet = featureSelection.relief(df_train, 1, 1000)
    #df_train, df_test, df_validation = apply_feature_selection(df_train, df_test, df_validation, featureSet)
    #print("Score for Relief: ")
    #print(featureSelection.getScore(df_test.iloc[:, 1:], df_test.iloc[:, 0]))

    featureSet = featureSelection.sfs(df_train)
    df_train, df_test, df_validation = apply_feature_selection(df_train, df_test, df_validation, featureSet)
    print("Score for SFS: ")
    print(featureSelection.getScore(df_test.iloc[:, 1:], df_test.iloc[:, 0]))

    
    #save_files(df_train, df_test, df_validation)

    # check accuracy with algorithms
    exploreData.check_accuracy_with_algorithms(df_test)


if __name__ == "__main__":
    main()