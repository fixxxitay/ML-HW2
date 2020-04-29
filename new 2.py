def remove_outliers(x_train: DataFrame, x_val: DataFrame, x_test: DataFrame,
                    z_threshold: float):
    mean_train = x_train[normal_features].mean()
    std_train = x_train[normal_features].std()

    dist_train = (x_train[normal_features] - mean_train) / std_train
    dist_val = (x_val[normal_features] - mean_train) / std_train
    dist_test = (x_test[normal_features] - mean_train) / std_train

    data_list = [x_train, x_val, x_test]
    dist_list = [dist_train, dist_val, dist_test]

    for feature in normal_features:
        for df, dist in zip(data_list, dist_list):
            for i in dist[feature].loc[(dist[feature] > z_threshold) | (dist[feature] < -z_threshold)].index:
                df.at[i, feature] = np.nan

    return x_train, x_val, x_test