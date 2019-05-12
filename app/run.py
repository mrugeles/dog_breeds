import json
import os
import numpy as np
from model_utillities import ModelUtilities
from prediction import Prediction

from werkzeug.utils import secure_filename
from flask import Flask
from flask import render_template, request, jsonify
from flask import redirect, url_for

from PIL import ImageFile


UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# init model
MODEL = 'model.hdf5'
model_utilities = ModelUtilities(MODEL)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    file_path = ''
    best_prediction_probability = None
    best_prediction_breed = None
    best_prediction = None
    breed_code = None
    predicted_result = None
    prediction = None
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            error = 'No file part'
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            error = 'No selected file'
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save('static/'+file_path)
            with model_utilities.graph.as_default():

                prediction, best_prediction_breed, breed_index = model_utilities.predict_image('static/'+file_path)

                if (breed_index < 10):
                    breed_code = '00'+str(breed_index)
                elif (breed_index > 9 and breed_index < 100):
                    breed_code = '0'+str(breed_index)
                else:
                    breed_code = str(breed_index)
                predicted_sample = 'static'

    return render_template('index.html', file_path = file_path, prediction = prediction, best_prediction_breed = best_prediction_breed, breed_code = breed_code)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
