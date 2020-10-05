import numpy as np
import pandas as pd

from itertools import compress

x = open('/home/blackrose/Desktop/algocon/ltp_dump_short.txt')
ticks = x.readlines()

#print(ticks[0])

ticks = list(compress(ticks, list(map(lambda x: 'NIFTY 50' in x, ticks))))

ltps = []
for tick in ticks:
    ltps.append(float(tick.split('ltp": ')[1].split(",")[0]))

#print(ltps)

ltps = pd.DataFrame(ltps)

ltps.columns = ['t']

ltps = ltps.diff().apply(lambda x: (np.sign(x)+1)/2)

ltps['t-1'] = ltps['t'].shift()

for n in range(2, 10):
    ltps[f't-{n}'] = ltps[f't-{n-1}'].shift()

ltps['t+1'] = ltps['t'].shift(-1)

ltps = ltps.dropna()

ltps = ltps.applymap(int)

#print(ltps.head(10))

X = ltps.iloc[:, :10].values
y = ltps.iloc[:, 10].values

from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm 

def optimize_RFC(X, y):

    for _ in tqdm(range(10)):

        best_model = None
        best_score = 0
        best_params = []

        n_estimators = np.random.randint(5, 500)
        min_sample_split = np.random.randint(2, 100)

        model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_sample_split)

        model.fit(X, y)
        score = model.score(X, y)

        if score > best_score:
            best_model = model
            best_score = score
            best_params = [n_estimators, min_sample_split]

    return best_model, best_score, best_params

# m, s, p = optimize_RFC(X, y)

benchmark_s = 0.72

# Monte Carlo Permutation Testing


fake_scores = []

for _ in range(10):

    X_fake = X.copy()

    np.random.shuffle(X_fake)

    m, s, p = optimize_RFC(X_fake, y)

    fake_scores.append(s)

    print(_, s)