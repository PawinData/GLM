import json
import pandas as pd
import numpy as np
from numpy.linalg import norm
from random import random
from scipy.sparse.linalg import svds
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer

k = 500

DATA = pd.read_csv("https://drive.google.com/uc?export=download&id=1GaZP3LrOY4F7VRfy6hSOoCNziUS81apf")
DATA = DATA.sort_values(by=["target"])
truth = DATA["target"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=100000, max_df=0.5, smooth_idf=True)
X = vectorizer.fit_transform(DATA["text"])
M = X.shape[0]  # the number of documents (Tweets)
N = X.shape[1]  # the number of terms

COUNTS = dict()
for sentiment in range(3):
    data = DATA.loc[DATA["target"]==sentiment]
    COUNTS[sentiment] = data.shape[0]
    
ACCUM = {-1:0, 0:COUNTS[0], 1:COUNTS[0]+COUNTS[1], 2:COUNTS[0]+COUNTS[1]+COUNTS[2]}

del DATA, vectorizer # remove redundant objects

U, Sigma, V_T = svds(X, k)
Sigma = np.diag(Sigma)
P = U.dot(np.diag([norm(V_T[j,:])**2 for j in range(k)]))    # new representation matrix (low-rank)

def obj_func(center, P, ACCUM, sentiment):
    d = 0
    for row in range(ACCUM[sentiment-1],ACCUM[sentiment]):
        v = P[row,:]
        d += v.dot(center) / (norm(v) * norm(center))
        return(-d)
        
iter_init = np.array([random() for ele in range(k)])
CENTERS = dict()
for sentiment in range(3):
    CENTERS[sentiment] = minimize(obj_func, x0=iter_init, args(P,ACCUM,sentiment))
    CENTERS[sentiment] = [float(element) for element in CENTERS[sentiment]]
    
with open("CENTERS.json", "w") as file:
    file.write(json.dumps(CENTERS, indent=4))
