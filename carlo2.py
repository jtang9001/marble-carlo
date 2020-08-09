
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial

rng = np.random.default_rng()

df = pd.read_csv('standings2.csv')
df = df.set_index("Team")

poss_scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25]

# ranks = pd.DataFrame(index = df.index)
# scores = pd.DataFrame(index = df.index)
# firstPlaceHistories = pd.DataFrame(index = df.index)

numColsBlank = len(df.columns[df.isna().any()])
numColsFilled = 16 - numColsBlank

# for colToIgnore in range(numColsFilled + 1):
#     firstPlaceHistories.loc[df.index[0], colToIgnore] = 0

# firstPlaceHistories = firstPlaceHistories.fillna(0)

def fillDfOneIteration(i):
    global df, numColsFilled

    if i % 100 == 0:
        print("Iteration", i)

    filledDf = df.copy()

    for col in filledDf.columns[numColsFilled:]:
        rng.shuffle(poss_scores)
        filledDf[col] = poss_scores
    newScore = filledDf.sum(axis=1)

    newFirstPlace = list(range(numColsFilled + 1))
    for colToIgnore in range(numColsFilled, -1, -1):
        rng.shuffle(poss_scores)
        filledDf.iloc[:, colToIgnore] = poss_scores
        scoresTemp = filledDf.sum(axis=1)
        bestTeam = scoresTemp.idxmax()
        newFirstPlace[colToIgnore] = bestTeam

    return newScore, newScore.rank(ascending = False), newFirstPlace

if __name__ == "__main__":
    with mp.Pool(processes = 8) as p:
        mp.freeze_support()

        iterations = 500000

        parallelResults = p.map(fillDfOneIteration, range(iterations), chunksize = 100)

        scores = pd.DataFrame(data = [x[0] for x in parallelResults]).transpose()
        # print(scores)

        ranks = pd.DataFrame(data = [x[1] for x in parallelResults]).transpose()
        # print(ranks)

        firstPlaces = pd.DataFrame(data = [x[2] for x in parallelResults])

        firstPlaceHistories = pd.DataFrame(index = df.index)
        for eventNum, col in firstPlaces.iteritems():
            firstPlaceHistories[eventNum] = col.value_counts(normalize = True)

        firstPlaceHistories = firstPlaceHistories.fillna(0)
        print(firstPlaceHistories)

        ranks.to_pickle("ranks.pkl")
        scores.to_pickle("scores.pkl")
        firstPlaceHistories.to_pickle("firstPlace.pkl")