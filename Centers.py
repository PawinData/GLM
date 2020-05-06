import numpy as np
import pandas as pd
import json
from numpy.linalg import norm
from scipy.sparse.linalg import svds
from scipy.optimize import minimize

k = 500


Airline = pd.read_csv("https://drive.google.com/uc?export=download&id=1HruH1PQRPPPwYvYvNZzjHpR1rZNFvUWN", encoding="utf-8")
Airline = Airline[["tweet_id", "airline_sentiment", "text"]]
Airline.airline_sentiment = Airline.airline_sentiment.replace({"negative":0, "positive":1, "neutral":2})
print("Airline Dataset loaded")


Y = load("Document_Term_Airline.npz")
print("Document Term matrix loaded.")

UU, SIGMA, VV_T = svds(Y,k)
print("SVD done")
PP = UU.dot(np.diag([norm(VV_T[j,:])**2 for j in range(k)]))
print("low-rank representation matrix calculated.")

with open("ID_train.json", "r") as fhand:
    ID_train = json.load(fhand)
print("ID_train loaded")

# compute Centroids as initial values to calculate Cosine-Similarity Centers
Centroids = dict()
counts = dict()
for id in ID_train:
    sentiment = Airline["airline_sentiment"][id]
    counts[sentiment] = counts.get(sentiment,0) + 1
    Centroids[sentiment] = Centroids.get(sentiment,0) + PP[id,:]
for sentiment in range(3):
    Centroids[sentiment] /= counts[sentiment]
print("Centroids calculated")

# split training set by 3 sentiment, only store ID
Train_ID = {0:[], 1:[], 2:[]}
for id in ID_train:
    sentiment = Airline["airline_sentiment"][id]
    Train_ID[sentiment].append(id)
print("3 lists of ID split by sentiment")


# input: array P, each row = low-rank representation of a document
# input: sentiment = all rows in P are labeled with
# input: arbitrary center = to be optimized with respect to
# output: sum of cosine similarity between center and all rows in P
def obj_f(center, P, sentiment):
    d = 0
    mg = norm(center)
    for row in range(P.shape[0]):
        v = P[row,:]
        d += v.dot(center) / (norm(v)*mg)
    return(-d / P.shape[0])
print("objective function defined")

Centers = dict()
for sentiment in range(3):
    print("start an optimization")
    result = minimize(obj_f, x0=Centroids[sentiment], args=(PP[Train_ID[sentiment,]], sentiment),)
    Centers[sentiment] = result.x
    Centers[sentiment] = [ele for ele in Centers[sentiment]]
    print("end an optimization")
    
json.dump(Centers, open("Centers.json", "w"))
