import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)

lrModel = pickle.load(open("pickleLrPredictionModel.pkl", "rb"))

@app.route('/', methods = ["GET","POST"])
def home():
    return render_template('index.html')

@app.route('/predict', methods = ["GET","POST"])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = lrModel.predict(final_features)

    output = np.round(prediction[0][0], 2)
    
    predictionText = "We suggest a price of $ {}".format(output)
    predictionText2 = "for the car ({}, {} km).".format(final_features[0][0],final_features[0][1])

    if output >= 100:
        return render_template('index.html', prediction_text=predictionText, prediction_text2 = predictionText2)
    else:
        return render_template('index.html', prediction_text='The car price is too low (< $ 100).', prediction_text2 = "We do not suggest the car to be sold.")

if __name__ == "__main__":
    app.run(debug=True)