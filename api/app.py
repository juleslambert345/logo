from flask import Flask, request, jsonify, send_file, render_template
import mlflow.pyfunc
import json
from PIL import Image
import io
import numpy as np
import torchvision.utils as vutils
import base64

# Name of the apps module package
#app = Flask(__name__)

app = Flask(__name__, template_folder='web_page_interaction')

# Load in the model at app startup
model = mlflow.pyfunc.load_model('./model/model_api')
import inspect
print(inspect.getfullargspec(model.predict))

# Load in our meta_data
f = open("./model/model_api/code/meta_data.txt", "r")
load_meta_data = json.loads(f.read())


# Meta data endpoint
@app.route('/', methods=['GET'])
def meta_data():
    return jsonify(load_meta_data)

@app.route('/logogan', methods=['POST', 'GET'])
def logogan_page():
    data_dic = {'type_prediction': 'random',
                'nb_logos': 1}
    image_file = model.predict(data_dic)
    image_file = base64.b64encode(image_file.getvalue())
    return render_template('logogan.html',result_image = image_file.decode('utf8'))

@app.route('/logogan/interpolation', methods=['POST', 'GET'])
def logogan_interpolation_page():

    if request.method == 'GET':
        logo1 = None
        logo2 = None
        logo_interpolation = None
    elif request.method == 'POST':
        logo1 = request.files['image1'].read()
        logo2 = request.files['image2'].read()



        logo1_pillow = Image.open(io.BytesIO(logo1))
        logo2_pillow = Image.open(io.BytesIO(logo2))
        data_dic = {'type_prediction':'encoding',
                    'img': logo1_pillow}
        z1 = model.predict(data_dic)
        data_dic = {'type_prediction': 'encoding',
                    'img': logo2_pillow}
        z2 = model.predict(data_dic)

        z = np.concatenate((z1, z2), axis=0)
        data_dic = {'type_prediction': 'interpolation_from_vector',
                    'nb_logos': 16,
                    'z': z}
        logo_interpolation = model.predict(data_dic)
        logo_interpolation = base64.b64encode(logo_interpolation.getvalue()).decode('utf8')

        logo1 = base64.b64encode(logo1)
        logo1 = logo1.decode('utf-8')
        logo2 = base64.b64encode(logo2)
        logo2 = logo2.decode('utf-8')

    #image1 = request.files['image1']
    return render_template('logogan_interpolation.html', logo1= logo1, logo2 = logo2, logo_interpolation= logo_interpolation)


@app.route('/logogan/starting', methods=['POST', 'GET'])
def logogan_starting_page():

    if request.method == 'GET':
        logo_input = None
        logo_output = None
    elif request.method == 'POST':
        logo_input = request.files['logo_input'].read()
        distance = request.form.get('distance', type = float)
        logo_pillow = Image.open(io.BytesIO(logo_input))
        data_dic = {'type_prediction': 'encoding',
                    'img': logo_pillow}
        z = model.predict(data_dic)
        z = z +distance*np.random.normal(0, 1, (16, z.shape[1]))

        data_dic = {'type_prediction': 'from_vector',
                    'z': z}
        logo_output = model.predict(data_dic)
        logo_output = base64.b64encode(logo_output.getvalue()).decode('utf8')

        logo_input = base64.b64encode(logo_input)
        logo_input = logo_input.decode('utf-8')

    return render_template('logogan_start.html', logo_input=logo_input, logo_output=logo_output)


# Prediction endpoint
@app.route('/predict', methods=['POST', 'GET'])
def predict_image():

    nb_logo = request.args.get('nb_logo', None) # to make it accessible in browser with the notation at the end : ?nb_logo=24
    if nb_logo is None:
        nb_logo = 64
    else:
        nb_logo = int(nb_logo)# because the arg of request is a string
    data_dic = {'type_prediction':'random',
                'nb_logos':nb_logo}
    image_file = model.predict(data_dic)
    return send_file(image_file, mimetype='image/PNG')

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)