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
model = mlflow.pyfunc.load_model('./experiments_result/2020_06_05__03_12_cluster_5')

# Load in our meta_data
f = open("../experiments_result/2020_06_05__03_12_cluster_5/code/meta_data.txt", "r")
load_meta_data = json.loads(f.read())


# Meta data endpoint
@app.route('/', methods=['GET'])
def meta_data():
    return jsonify(load_meta_data)

@app.route('/logogan', methods=['POST', 'GET'])
def hello1():
    result_image = get_image()
    return render_template('web_page_interaction.html',result_image = result_image.decode('utf8'))

def get_image():
    fakeimage = model.predict(0)

    # print('\n\tfakeimg = ',type(fakeimage))
    # print('\n\tfakeimg.shape = ',fakeimage.shape)

    imggrd = vutils.make_grid(fakeimage, padding=2, normalize=True)
    imgnp = imggrd.numpy()
    imgnpT = np.transpose(imgnp, (1, 2, 0))

    # print('\ntype(imgnp) =  ',type(imgnp))
    # print('\nshape(imgnp) = ',imgnp.shape) #(64, 3, 64, 64)

    imgPIL = Image.fromarray((255 * imgnpT).astype("uint8"), 'RGB')

    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    imgPIL.save(file_object, 'PNG')

    file_object.seek(0)

    file_object = base64.b64encode(file_object.getvalue())

    result_image = file_object
    # Return prediction as response
    # return jsonify(['DONE : Ideally this should be an image'])
    #return file_object


# Prediction endpoint
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    req = request.get_json()

    fakeimage = model.predict(0)

    # print('\n\tfakeimg = ',type(fakeimage))
    # print('\n\tfakeimg.shape = ',fakeimage.shape)

    imggrd = vutils.make_grid(fakeimage, padding=2, normalize=True)
    imgnp = imggrd.numpy()
    imgnpT = np.transpose(imgnp, (1, 2, 0))

    # print('\ntype(imgnp) =  ',type(imgnp))
    # print('\nshape(imgnp) = ',imgnp.shape) #(64, 3, 64, 64)

    imgPIL = Image.fromarray((255 * imgnpT).astype("uint8"), 'RGB')

    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    imgPIL.save(file_object, 'PNG')

    file_object.seek(0)

    # Return prediction as response
    # return jsonify(['DONE : Ideally this should be an image'])
    return send_file(file_object, mimetype='image/PNG')


app.run( port=5000, debug=True)