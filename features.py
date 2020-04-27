import pandas as pd
import numpy as np
import sklearn

def relief(df: pd.DataFrame, thresh: float, nbIterations: int):
    X, Y = df.values[:, 1:], df.values[:, 0]
    X = sklearn.preprocessing.MinMaxScaler().fit_transform(X)

    W = np.zeros(X.shape[1])

    for i in range(nbIterations):
        index = np.random.randint(low=0, high=X.shape[0])
        Xi = X[index, :]
        Yi = Y[index]

        hitGroup = X[Y == Yi, :]
        missGroup = X[Y != Yi,:]

        
        idHit = (hitGroup - Xi).sum(axis=0).argmin(axis=0)
        idMiss = (missGroup - Xi).sum(axis=0).argmin(axis=0)

        nearHit = hitGroup[idHit, :]
        nearMiss = missGroup[idMiss, :]

        W = W - ((Xi - nearHit) ** 2) + ((Xi - nearMiss) ** 2)


    ret = np.zeros(X.shape[1])
    ret[W > thresh] = 1
    
    return ret


def getScore(X, Y):
    knnModel = sklearn.neighbors.KNeighborsClassifier()
    
    return sklearn.model_selection.cross_val_score(knnModel, X, Y, scoring='accuracy').mean()


def sfs(df: pd.DataFrame):
    X, Y = df.iloc[:, 1:], df.iloc[:, 0]
    globalBestScore = np.NINF
    selectedFeatures = list()
    remainingFeatures = list(X.columns)
    

    for i in range(len(X.columns)):
        bestScore = np.NINF
        bestFeature = None
        for feature in remainingFeatures:
            score = getScore(X[selectedFeatures + [feature]], Y)

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

