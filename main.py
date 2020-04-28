import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from exploreData import explore_data
from features import nominal_features, integer_features, float_features


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


def save_files(df: pd.DataFrame):
    # Save the transformed data
    df_train = df.iloc[0:round(len(df) * 0.6), :]
    df_test = df.iloc[round(len(df) * 0.6):round(len(df) * 0.8), :]
    df_validation = df.iloc[round(len(df) * 0.8):len(df), :]

    df_train.to_csv('train.csv', index=False)
    df_test.to_csv('test.csv', index=False)
    df_validation.to_csv('validation.csv', index=False)


def apply_filter(df: pd.DataFrame):
    # The filter method : correlation factor between features
    # Remove the highly correlated ones
    correlated_features = set()
    correlation_matrix = df.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
  
    df.drop(labels=correlated_features, axis=1, inplace=True)

    return df


def apply_wrapper(df: pd.DataFrame):
    # Wrapper method :
    model = LogisticRegression()
    rfe = RFE(model, 16)
    fit = rfe.fit(df.values[:, 1:], df.values[:, 0])
    #print("Num Features: %s" % (fit.n_features_))
    #print("Selected Features: %s" % (fit.support_))
    #print("Feature Ranking: %s" % (fit.ranking_))

    array_bool = np.array([True], dtype=bool)
    array_bool = np.append(array_bool, fit.support_)
    df = df.iloc[:, array_bool]

    return df


def remove_wrong_party(df: pd.DataFrame):
    df = df[df.Vote != 10]
    df = df[df.Vote != 4]
    df = df.dropna()
    return df


def save_raw_data(df_test, df_train, df_validation):
    df_train.to_csv('raw_train.csv', index=False)
    df_test.to_csv('raw_test.csv', index=False)
    df_validation.to_csv('raw_validation.csv', index=False)


def complete_missing_values(df: pd.DataFrame):
    df = df[df > 0]
    # Fill missing values with the most common value per column
    df.loc[:, nominal_features] = df.apply(lambda x: x.fillna(x.mode()[0]), axis=0)
    # Fill by mean rounded to nearest integer
    df.loc[:, integer_features] = df.apply(lambda x: x.fillna(round(x.mean())), axis=0)
    # Fill by mean
    df.loc[:, float_features] = df.apply(lambda x: x.fillna(x.mean()), axis=0)
    return df


def nominal_to_numerical_categories(df: pd.DataFrame):
    # from nominal to Categorical
    df = df.apply(lambda x: pd.Categorical(x) if x.dtype != 'float64' else x, axis=0)
    # give number to each Categorical
    df = df.apply(lambda x: x.cat.codes if x.dtype != 'float64' else x, axis=0)
    return df


def main():
    df = pd.read_csv("ElectionsData.csv")

    print(df.isnull().sum())

    # # split the data to train , test and validation
    # df_train, df_test, df_validation = train_test_validation_split(df, 0.2, 0.2)
    #
    # # Save the raw data first
    # save_raw_data(df_test, df_train, df_validation)
    #
    # # Convert nominal types to numerical categories
    # df = nominal_to_numerical_categories(df)
    #
    # # 1 - Imputation - Complete missing values
    # df = complete_missing_values(df)
    #
    # # Remove lines with wrong party (Violets | Khakis)
    # df = remove_wrong_party(df)
    #
    # # 2 - Data Cleansing
    # # Outlier detection using z score
    #
    # # z = np.abs(stats.zscore(df))
    # # df = df[(z < 3).all(axis=1)]
    #
    # # 3 - Normalization (scaling)
    #
    # # print some graph about the data
    # explore_data(df)
    #
    # # 4 - Feature Selection
    # df = apply_filter(df)
    # df = apply_wrapper(df)
    #
    # print(df.columns.values)
    #
    #
    #
    # # save files
    # save_files(df)

    # check accuracy with algorithms
    # check_accuracy_with_algorithms(df)


if __name__ == "__main__":
    main()