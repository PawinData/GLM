import json
import numpy as np
import pandas as pd
from random import sample
from scipy.optimize import differential_evolution
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds

k = 500

Airline = pd.read_csv("https://drive.google.com/uc?export=download&id=1HruH1PQRPPPwYvYvNZzjHpR1rZNFvUWN", encoding="utf-8")

Airline = Airline[["tweet_id", "airline_sentiment", "text"]]
Airline.airline_sentiment = Airline.airline_sentiment.replace({"negative":0, "positive":1, "neutral":2})

ID_train = json.load(open("ID_train.json"))  # load ID of training set
Train = Airline.iloc[ID_train]

Train = Train.sort_values(by=["airline_sentiment"])
truth = Train["airline_sentiment"].tolist()

vectorizer = TfidfVectorizer(stop_words="english", max_features=100000, max_df=0.5, smooth_idf=True)
Y = vectorizer.fit_transform(Train["text"])
M,N = Y.shape

counts = dict()
for sentiment in range(3):
    data = Train.loc[Train["airline_sentiment"]==sentiment]
    counts[sentiment] = data.shape[0]
print(counts)

accum = {-1:0, 0:counts[0], 1:counts[0]+counts[1], 2:counts[0]+counts[1]+counts[2]}

UU, SIGMA, VV_T = svds(Y, k)
SIGMA = np.diag(SIGMA)
PP = UU.dot(np.diag([norm(VV_T[j,:])**2 for j in range(k)]))

Centers = dict()
Bounds = [(min(PP[:,j]), max(PP[:,j])) for j in range(k)]
for sentiment in range(3):
    res = differential_evolution(obj_func, bounds=Bounds, args=(PP,accum,sentiment), maxiter=100000)
    Centers[sentiment] = res.x 
    # numpy arrays --> lists for storing in json file
    Centers[sentiment] = [float(ele) for ele in Centers[sentiment]]  
    
with open("Centers.json", "w") as file:
    file.write(json.dumps(Centers, indent=4))
