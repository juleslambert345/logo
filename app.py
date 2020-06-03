from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import json
from os.path import join

experiment_path = join('experiments_result', '2020_05_31__19_53_test3')

# Name of the apps module package
app = Flask(__name__)

# Load in the model at app startup
model = mlflow.pyfunc.load_model(experiment_path)

# Load in our meta_data
f = open(join(experiment_path, 'code', 'meta_data.txt'), "r")
load_meta_data = json.loads(f.read())


# Meta data endpoint
@app.route('/', methods=['GET'])
def meta_data():
    return jsonify(load_meta_data)


# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()

    # Log the request
    print({'request': req})


    # Format the request data in a DataFrame
    nb_logo = req['nb_logo']

    # Get model prediction - convert from np to list
    pred = model.predict(nb_logo)

    # Log the prediction
    print({'response': pred})

    # Return prediction as reponse
    return jsonify(pred)

app.run(  debug=True)