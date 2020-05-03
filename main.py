﻿import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from exploreData import explore_data
from features import nominal_features, integer_features, float_features, uniform_features, normal_features
import featureSelection
import exploreData
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import VarianceThreshold


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


def get_filter_selection(df_train: pd.DataFrame):
    var_filter = VarianceThreshold(threshold=0.05)
   
    var_filter.fit(df_train.iloc[:, 1:], df_train.iloc[:, 0])
   
    ret = np.zeros(df_train.iloc[:, 1:].shape[1])
    ret[var_filter.get_support(indices=True)] = 1

    return  ret


def get_wrapper_selection(df_train: pd.DataFrame):
    # Wrapper method :
    model = GradientBoostingClassifier(n_estimators=100, random_state=0)
    rfe = RFE(model, 19)
    fit = rfe.fit(df_train.values[:, 1:], df_train.values[:, 0])
    print("wrapper score is: ")
    print(fit.score)
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


def complete_missing_values(df_train: pd.DataFrame, df_test: pd.DataFrame, df_validation: pd.DataFrame)\
                                                                -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
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


def nominal_to_numerical_categories(df: pd.DataFrame):
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


def normalize(df_test: pd.DataFrame, df_train: pd.DataFrame, df_validation: pd.DataFrame):
    # min-max for uniform features
    uniform_scaler = MinMaxScaler(feature_range=(-1, 1))
    df_train[uniform_features] = uniform_scaler.fit_transform(df_train[uniform_features])
    df_validation[uniform_features] = uniform_scaler.transform(df_validation[uniform_features])
    df_test[uniform_features] = uniform_scaler.transform(df_test[uniform_features])

    # z-score for normal features
    normal_scaler = StandardScaler()
    df_train[normal_features] = normal_scaler.fit_transform(df_train[normal_features])
    df_validation[normal_features] = normal_scaler.transform(df_validation[normal_features])
    df_test[normal_features] = normal_scaler.transform(df_test[normal_features])
    return df_train, df_test, df_validation


def remove_outliers(threshold: float, df_train: pd.DataFrame, df_validation: pd.DataFrame, df_test: pd.DataFrame):
    mean = df_train[normal_features].mean()
    std = df_train[normal_features].std()

    z_train = (df_train[normal_features] - mean) / std
    z_val = (df_validation[normal_features] - mean) / std
    z_test = (df_test[normal_features] - mean) / std

    df_train[z_train.mask(abs(z_train) > threshold).isna()] = np.nan
    df_validation[z_val.mask(abs(z_val) > threshold).isna()] = np.nan
    df_test[z_test.mask(abs(z_test) > threshold).isna()] = np.nan

    return df_train, df_validation, df_test


def sbs_function(df_train: pd.DataFrame):
    knn = KNeighborsClassifier(n_neighbors=3)
    # clf_gradient = GradientBoostingClassifier(n_estimators=100, random_state=0)
    sbs = SFS(knn,
              k_features=8,
              forward=False,  # if forward = True then SFS otherwise SBS
              floating=False,
              verbose=2,
              n_jobs=-1,
              scoring='accuracy')
    # after applying sfs fit the data:
    sbs.fit(df_train.values[:, 1:], df_train.values[:, 0])

    # to get the final set of features

    ret = np.zeros(df_train.shape[1])
    for index in sbs.k_feature_idx_:
            ret[index] = 1

    array_bool = np.array(ret, dtype=bool)

    return array_bool


def main():
    df = pd.read_csv("ElectionsData.csv")

    all_features_list = df.columns.values

    # Convert nominal types to numerical categories
    df = nominal_to_numerical_categories(df)

    # split the data to train , test and validation
    df_train, df_test, df_validation = deterministicSplit(df, 0.6, 0.2) # train_test_validation_split(df, 0.2, 0.2)

    # Save the raw data first
    save_raw_data(df_test, df_train, df_validation)

    # 1 - Imputation - Complete missing values
    df_train, df_test, df_validation = complete_missing_values(df_train, df_test, df_validation)

    # 2 - Data Cleansing
    # Outlier detection using z score
    threshold = 3#.3
    df_train, df_validation, df_test = remove_outliers(threshold, df_train, df_validation, df_test)

    # Remove lines with wrong party (Violets | Khakis)
    df_train, df_test, df_validation = remove_wrong_party_and_na(df_train, df_test, df_validation)

    # 3 - Normalization (scaling)
    df_train, df_test, df_validation = normalize(df_test, df_train, df_validation)

    # print some graph about the data
    #explore_data(df)

    # 4 - Feature Selection
    featureSet = get_filter_selection(df_train)
    df_train, df_test, df_validation = apply_feature_selection(df_train, df_test, df_validation, featureSet)

    model = KNeighborsClassifier()
    featureSet1 = featureSelection.relief(df_train, 2000, 60)
    #df_train, df_test, df_validation = apply_feature_selection(df_train, df_test, df_validation, featureSet1)
    #print("Score after Relief: ")
    #print(featureSelection.getScore(df_test.iloc[:, 1:], df_test.iloc[:, 0],model))

    #featureSet = sbs_function(df_train)
    model = KNeighborsClassifier()
    featureSet2 = featureSelection.sfs(df_train, model)
    #df_train, df_test, df_validation = apply_feature_selection(df_train, df_test, df_validation, featureSet2)
    #print("Score after SFS: ")
    #print(featureSelection.getScore(df_test.iloc[:, 1:], df_test.iloc[:, 0],model))


    model = KNeighborsClassifier()
    featureSet = featureSet1 + featureSet2
    featureSet[featureSet > 0] = True
    featureSet[featureSet <= 0] = False
    df_train, df_test, df_validation = apply_feature_selection(df_train, df_test, df_validation, featureSet)
    print(featureSelection.getScore(df_test.iloc[:, 1:], df_test.iloc[:, 0], model))


    save_files(df_train, df_test, df_validation)

    # check accuracy with algorithms
    #exploreData.check_accuracy_with_algorithms(df_train, df_validation)
    print(df_train.columns.values)


if __name__ == "__main__":
    main()