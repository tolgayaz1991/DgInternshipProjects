import pandas as pd

import pickle

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

simpleHyundaiPrice = pd.read_csv("simpleHyundaiPrice.csv")

X = simpleHyundaiPrice[["year", "km_driven"]]

y = simpleHyundaiPrice [["selling_price"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 0)

lrm = LinearRegression()

lrm.fit(X_train, y_train)


with open("pickleLrPredictionModel.pkl", "wb") as f:
    pickle.dump(lrm, f)
    
    
# try the model
with open("pickleLrPredictionModel.pkl", "rb") as f:
    lrModel = pickle.load(f)

print("{}".format(lrModel.predict(X_test)))