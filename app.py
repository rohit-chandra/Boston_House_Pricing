
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, app, jsonify, url_for, render_template

app = Flask(__name__)

# Losad the model
regression_model = pickle.load(open("Boston_regression.pkl", "rb"))
std_scaler = pickle.load(open("scaling.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods = ["POST"])
def predict_api():
    data = request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    
    # scale the new data
    new_tranformed_data = std_scaler.transform(np.array(list(data.values())).reshape(1, -1))
    
    # prediction
    output = regression_model.predict(new_tranformed_data)
    # it's a 2D array
    print(output[0])
    
    return jsonify(output[0])
    

@app.route("/predict", methods = ["POST"])
def predict():
    """
    predict the price from the HTML page
    
    """
    # convert to float
    # capture all the values from the HTML form
    data = [float(x) for x in request.form.values()]
    # perform scaling for those captured values
    final_input = std_scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    # prediction
    output = regression_model.predict(final_input)[0]
    
    return render_template("home.html", prediction_text = f"The House Price Prediction is =  {output}")
    
    
    
    
    

if __name__ == "__main__":
    app.run(debug=True)
