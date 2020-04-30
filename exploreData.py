from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import Perceptron
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler


def calc_err(y_pred, y_true):
    error = 1-np.mean(y_pred == y_true)
    return error


def train_labels_validation_labels(df: pd.DataFrame):
    df_train = df.iloc[0:round(len(df) * 0.6), :]
    df_validation = df.iloc[round(len(df) * 0.8):len(df), :]
    train_without_label = df_train.values[:, 1:]
    train_labels = df_train.values[:, 0]
    validation_without_label = df_validation.values[:, 1:]
    validation_labels = df_validation.values[:, 0]
    return train_labels, train_without_label, validation_labels, validation_without_label


def evaluate_classifier(clf, df_train: pd.DataFrame, df_validation: pd.DataFrame, num_repeats=10):
    current_errors = np.zeros(num_repeats)
    train_labels, train_without_label, = df_train.values[:, 0], df_train.values[:, 1:]
    validation_labels, validation_without_label = df_validation.values[:, 0], df_validation.values[:, 1:]
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


def knn_l1_l2_cosine(df_train: pd.DataFrame, df_validation: pd.DataFrame):
    k = 5
    clf2 = KNeighborsClassifier(n_neighbors=k, p=2)
    l2_error, l2_error_std = evaluate_classifier(clf2, df_train, df_validation)
    print("\nclassification error for KNeighborsClassifier with l2: {} ({}%)".format(l2_error, l2_error * 100))
    clf1 = KNeighborsClassifier(n_neighbors=k, p=1)
    l1_error, l1_error_std = evaluate_classifier(clf1, df_train, df_validation)
    print("\nclassification error for KNeighborsClassifier with l1: {} ({}%)".format(l1_error, l1_error * 100))
    clf_cosine = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    cos_error, cos_error_std = evaluate_classifier(clf_cosine, df_train, df_validation)
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


def check_best_k_for_knn(df_train: pd.DataFrame, df_validation: pd.DataFrame):
    k_s = [1, 3, 5, 7, 15, 20, 25]  # num neighbors
    # using distances:
    # Cosine Distance - metric='cosine' [KNeighborsClassifier(n_neighbors=K, metric='cosine')]
    k_errors = np.zeros(len(k_s))
    k_errors_std = np.zeros(len(k_s))
    for index, K in enumerate(k_s):
        print("K: {}".format(K))
        clf_cosine = KNeighborsClassifier(n_neighbors=K, metric='cosine')
        error_mean, error_std = evaluate_classifier(clf_cosine, df_train, df_validation)
        k_errors[index] = np.mean(error_mean)
        k_errors_std[index] = np.std(error_std)
    print_knn_k_graph(k_errors, k_errors_std, k_s)


def check_perceptron(df_train: pd.DataFrame, df_validation: pd.DataFrame):
    data_without_label = df_train.values[:, 1:]
    data_labels = df_train.values[:, 0]
    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(data_without_label, data_labels)
    error_mean = 1 - clf.score(data_without_label, data_labels)
    print("\nclassification error for Perceptron: {} ({}%)".format(error_mean, error_mean * 100))


def check_accuracy_with_algorithms(df_train: pd.DataFrame, df_validation: pd.DataFrame):
    # GaussianNB()
    error_mean, error_std = evaluate_classifier(GaussianNB(), df_train, df_validation)
    print("\nclassification error for GaussianNB: {} ({}%)".format(error_mean, error_mean * 100))

    # KNeighborsClassifier()
    knn_l1_l2_cosine(df_train, df_validation)

    # KNeighborsClassifier() performance vs. K
    check_best_k_for_knn(df_train, df_validation)

    # Perceptron
    check_perceptron(df_train, df_validation)

    clf_gradient = GradientBoostingClassifier(n_estimators=100, random_state=0)
    error_mean, error_std = evaluate_classifier(clf_gradient, df_train, df_validation)
    print("\nclassification error for GradientBoostingClassifier: {} ({}%)".format(error_mean, error_mean * 100))




def features_histograms(df: pd.DataFrame):
    plt.close('all')
    all_features_ = list(df.keys())

    for feature_ in all_features_:
        plt.title(feature_)
        plt.hist(df[feature_].values)
        plt.savefig("graphs/{}.png".format(feature_))
        plt.show()


def feature_label_relationship(df: pd.DataFrame):
    plt.close('all')
    y_df_ = df['Vote']
    x_df_ = df.drop('Vote', axis=1)
    all_features_ = sorted(list(x_df_.keys()))
    print("l_map: {}".format({'Blues': 0, 'Browns': 1, 'Greens': 2, 'Greys': 3, 'Khakis': 4, 'Oranges': 5, 'Pinks': 6,
                              'Purples': 7, 'Reds': 8, 'Turquoises': 9, 'Violets': 10, 'Whites': 11, 'Yellows': 12}))

    for feature_ in all_features_:
        print(feature_)
        plt.scatter(x_df_[feature_], y_df_)
        plt.xlabel(feature_)
        plt.ylabel('Vote', color='g')
        plt.savefig("graphs/{}.png".format('Vote' + "_Vs._" + feature_))
        plt.show()


def explore_data(df):
    feature_label_relationship(df)
    features_histograms(df)