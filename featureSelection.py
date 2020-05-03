import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

def Nmaxelements(list1, N):
    final_list = []

    for i in range(0, N):
        max1 = 0

        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j];

        list1.remove(max1);
        final_list.append(max1)
    return final_list


def relief(df: pd.DataFrame, nbIterations: int, n: int):
    X, Y = df.values[:, 1:], df.values[:, 0]
    X = sklearn.preprocessing.MinMaxScaler().fit_transform(X)

    W = np.zeros(X.shape[1])

    for i in range(nbIterations):
        index = np.random.randint(low=0, high=X.shape[0])
        Xi = X[index, :]
        Yi = Y[index]

        hitGroup = X[Y == Yi, :]
        missGroup = X[Y != Yi, :]

        idHit = (hitGroup - Xi).sum(axis=0).argmin(axis=0)
        idMiss = (missGroup - Xi).sum(axis=0).argmin(axis=0)

        nearHit = hitGroup[idHit, :]
        nearMiss = missGroup[idMiss, :]

        W = W - ((Xi - nearHit) ** 2) + ((Xi - nearMiss) ** 2)

    ret = np.zeros(X.shape[1])

    ret = np.zeros(X.shape[1])
    ret[W > n] = 1

    #list_ = W.tolist()
    #highest_list = Nmaxelements(list_, n)
    #print("highest_list is: ")
    #print(highest_list)

    #print("W is ")
    #print(W)
    #for i in range(0, X.shape[1]):
    #    if W[i] in highest_list:
    #        ret[i] = 1
    #print("ret is ")
    #print(ret)
    return ret


def getScore(X, Y, model):
    #cv = ShuffleSplit()
    return sklearn.model_selection.cross_val_score(model, X, Y, scoring='accuracy').mean()#, cv=cv).mean()


def sfs(df: pd.DataFrame, model):
    X, Y = df.iloc[:, 1:], df.iloc[:, 0]
    globalBestScore = np.NINF
    selectedFeatures = list()
    remainingFeatures = list(X.columns)

    for i in range(len(X.columns)):
        bestScore = np.NINF
        bestFeature = None
        for feature in remainingFeatures:
            score = getScore(X[selectedFeatures + [feature]], Y, model)

            if score > bestScore:
                bestScore = score
                bestFeature = feature

        if bestScore > globalBestScore:
            globalBestScore = bestScore
            selectedFeatures.append(bestFeature)
            remainingFeatures.remove(bestFeature)

        else:
            break

    ret = np.zeros(X.shape[1])

    for i in range(len(ret)):
        if X.columns[i] in selectedFeatures:
            ret[i] = 1

    return ret