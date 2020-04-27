import matplotlib.pyplot as plt
import pandas as pd


def features_histograms(df: pd.DataFrame):
    plt.close('all')
    local_all_features = list(df.keys())
    for f in local_all_features:
        plt.title(f)
        plt.hist(df[f].values)
        plt.savefig("graphs/{}.png".format(f))
        plt.show()


def feature_label_relationship(df: pd.DataFrame):

    # categorized nominal attributes to int
    y_df = df['Vote']
    x_df = df.drop('Vote', axis=1)
    plt.close('all')
    local_all_features = sorted(list(x_df.keys()))
    print("label map: {}".format({'Blues': 0, 'Browns': 1, 'Greens': 2, 'Greys': 3, 'Khakis': 4, 'Oranges': 5, 'Pinks': 6,
                                 'Purples': 7, 'Reds': 8, 'Turquoises': 9, 'Violets': 10, 'Whites': 11, 'Yellows': 12}))
    for f in local_all_features:
        print(f)
        plt.scatter(x_df[f], y_df)
        plt.ylabel('Vote', color='b')
        plt.xlabel(f)
        plt.savefig("graphs/{}.png".format('Vote' + "_Vs._" + f))
        plt.show()


def explore_data(df):
    feature_label_relationship(df)
    features_histograms(df)