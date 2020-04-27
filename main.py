import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt

from exploreData import explore_data
from features import nominal_features, integer_features, float_features
from sklearn.naive_bayes import GaussianNB
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import Perceptron


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


def save_files(df):
    # Save the transformed data
    df_train = df.iloc[0:round(len(df) * 0.6), :]
    df_test = df.iloc[round(len(df) * 0.6):round(len(df) * 0.8), :]
    df_validation = df.iloc[round(len(df) * 0.8):len(df), :]

    df_train.to_csv('train.csv', index=False)
    df_test.to_csv('test.csv', index=False)
    df_validation.to_csv('validation.csv', index=False)


def wrapper_method(df):
    model = LogisticRegression()
    rfe = RFE(model, 16)
    data_without_label = df.values[:, 1:]
    labels = df.values[:, 0]
    fit = rfe.fit(data_without_label, labels)
    # print("Num Features: %s" % (fit.n_features_))
    # print("Selected Features: %s" % (fit.support_))
    # print("Feature Ranking: %s" % fit.ranking_)
    array_bool = np.array([True], dtype=bool)
    array_bool = np.append(array_bool, fit.support_)
    df = df.iloc[:, array_bool]
    # print(df.columns.values)
    return df


def feature_selection(df):
    # The filter method : correlation factor between features
    # Remove the highly correlated ones
    correlated_features = set()
    correlation_matrix = df.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                col_name = correlation_matrix.columns[i]
                correlated_features.add(col_name)
    df.drop(labels=correlated_features, axis=1, inplace=True)
    # print(df.columns.values)


def remove_wrong_party(df):
    df = df[df.Vote != 10]
    df = df[df.Vote != 4]
    df = df.dropna()
    return df


def save_raw_data(df_test, df_train, df_validation):
    df_train.to_csv('raw_train.csv', index=False)
    df_test.to_csv('raw_test.csv', index=False)
    df_validation.to_csv('raw_validation.csv', index=False)


def complete_missing_values(df):
    df = df[df > 0]
    # Fill missing values with the most common value per column
    df.loc[:, nominal_features] = df.apply(lambda x: x.fillna(x.mode()[0]), axis=0)
    # Fill by mean rounded to nearest integer
    df.loc[:, integer_features] = df.apply(lambda x: x.fillna(round(x.mean())), axis=0)
    # Fill by mean
    df.loc[:, float_features] = df.apply(lambda x: x.fillna(x.mean()), axis=0)
    return df


def nominal_to_numerical_categories(df):
    # from nominal to Categorical
    df = df.apply(lambda x: pd.Categorical(x) if x.dtype != 'float64' else x, axis=0)
    # give number to each Categorical
    df = df.apply(lambda x: x.cat.codes if x.dtype != 'float64' else x, axis=0)
    return df


def calc_err(y_pred, y_true):
    error = 1-np.mean(y_pred == y_true)
    return error


def train_labels_validation_labels(df):
    df_train = df.iloc[0:round(len(df) * 0.6), :]
    df_validation = df.iloc[round(len(df) * 0.8):len(df), :]
    train_without_label = df_train.values[:, 1:]
    train_labels = df_train.values[:, 0]
    validation_without_label = df_validation.values[:, 1:]
    validation_labels = df_validation.values[:, 0]
    return train_labels, train_without_label, validation_labels, validation_without_label


def evaluate_classifier(clf, df, num_repeats=10):
    current_errors = np.zeros(num_repeats)
    train_labels, train_without_label, validation_labels, validation_without_label = train_labels_validation_labels(df)
    for i_rep in tqdm(range(num_repeats)):
        # train
        clf.fit(train_without_label, train_labels)
        # test
        y_pred = clf.predict(validation_without_label)
        # calculate error
        current_errors[i_rep] = calc_err(y_pred, validation_labels)

    error_mean = np.mean(current_errors)
    error_std = np.std(current_errors)

    return error_mean, error_std


def print_summary_knn_l1_l2_cosine(cos_error, cos_error_std, l1_error, l1_error_std, l2_error, l2_error_std):
    summary_df = pd.DataFrame(np.concatenate([np.array([l2_error, l1_error, cos_error]).reshape(-1, 1),
                                              np.array([1 - l2_error, 1 - l1_error, 1 - cos_error]).reshape(-1, 1),
                                              np.array([l2_error_std, l1_error_std, cos_error_std]).reshape(-1, 1)],
                                             axis=1),
                              columns=['Error', 'Accuracy', 'Error STD'], index=['L2', 'L1', 'Cosine'])
    print(summary_df)


def knn_l1_l2_cosine(df):
    k = 5
    clf2 = KNeighborsClassifier(n_neighbors=k, p=2)
    l2_error, l2_error_std = evaluate_classifier(clf2, df)
    print("\nclassification error for KNeighborsClassifier with l2: {} ({}%)".format(l2_error, l2_error * 100))
    clf1 = KNeighborsClassifier(n_neighbors=k, p=1)
    l1_error, l1_error_std = evaluate_classifier(clf1, df)
    print("\nclassification error for KNeighborsClassifier with l1: {} ({}%)".format(l1_error, l1_error * 100))
    clf_cosine = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    cos_error, cos_error_std = evaluate_classifier(clf_cosine, df)
    print("\nclassification error for KNeighborsClassifier with cosine: {} ({}%)".format(cos_error, cos_error * 100))
    print_summary_knn_l1_l2_cosine(cos_error, cos_error_std, l1_error, l1_error_std, l2_error, l2_error_std)


def print_knn_k_graph(k_errors, k_errors_std, k_s):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.errorbar(k_s, k_errors, yerr=k_errors_std, uplims=True, lolims=True)
    ax.set_xlabel("number of neighbors (K)")
    ax.set_ylabel("error %")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()
    ax.set_title("Test Error vs. Number of Neighbors (N={} Repeats)".format(10))
    plt.show()


def check_best_k_for_knn(df):
    k_s = [1, 3, 5, 7, 15, 20, 25]  # num neighbors
    # using distances:
    # Cosine Distance - metric='cosine' [KNeighborsClassifier(n_neighbors=K, metric='cosine')]
    k_errors = np.zeros(len(k_s))
    k_errors_std = np.zeros(len(k_s))
    for index, K in enumerate(k_s):
        print("K: {}".format(K))
        clf_cosine = KNeighborsClassifier(n_neighbors=K, metric='cosine')
        error_mean, error_std = evaluate_classifier(clf_cosine, df)
        k_errors[index] = np.mean(error_mean)
        k_errors_std[index] = np.std(error_std)
    print_knn_k_graph(k_errors, k_errors_std, k_s)


def check_perceptron(df):
    data_without_label = df.values[:, 1:]
    data_labels = df.values[:, 0]
    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(data_without_label, data_labels)
    error_mean = 1 - clf.score(data_without_label, data_labels)
    print("\nclassification error for Perceptron: {} ({}%)".format(error_mean, error_mean * 100))


def check_accuracy_with_algorithms(df):
    # GaussianNB()
    error_mean, error_std = evaluate_classifier(GaussianNB(), df)
    print("\nclassification error for GaussianNB: {} ({}%)".format(error_mean, error_mean * 100))

    # KNeighborsClassifier()
    knn_l1_l2_cosine(df)

    # KNeighborsClassifier() performance vs. K
    check_best_k_for_knn(df)

    # Perceptron
    check_perceptron(df)


def main():
    df = pd.read_csv("ElectionsData.csv")

    # split the data to train , test and validation
    df_train, df_test, df_validation = train_test_validation_split(df, 0.2, 0.2)

    # Save the raw data first
    save_raw_data(df_test, df_train, df_validation)

    # Convert nominal types to numerical categories
    df = nominal_to_numerical_categories(df)

    # 1 - Imputation - Complete missing values
    df = complete_missing_values(df)

    # Remove lines with wrong party (Violets | Khakis)
    df = remove_wrong_party(df)

    # 2 - Data Cleansing
    # Outlier detection using z score

    # z = np.abs(stats.zscore(df))
    # df = df[(z < 3).all(axis=1)]

    # 3 - Normalization (scaling)

    # print some graph about the data
    explore_data(df)

    # 4 - Feature Selection
    feature_selection(df)

    # Wrapper method :
    df = wrapper_method(df)

    # save files
    save_files(df)

    # check accuracy with algorithms
    # check_accuracy_with_algorithms(df)


if __name__ == "__main__":
    main()