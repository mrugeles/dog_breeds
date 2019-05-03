import json
import os
import numpy as np

from werkzeug.utils import secure_filename

from flask import Flask
from flask import render_template, request, jsonify
from flask import redirect, url_for

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing import image
from keras.models import model_from_json

import tensorflow as tf

from PIL import ImageFile

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
DOG_NAMES =['001.Affenpinscher', '002.Afghan_hound', '003.Airedale_terrier', '004.Akita', '005.Alaskan_malamute', '006.American_eskimo_dog', '007.American_foxhound', '008.American_staffordshire_terrier', '009.American_water_spaniel', '010.Anatolian_shepherd_dog', '011.Australian_cattle_dog', '012.Australian_shepherd', '013.Australian_terrier', '014.Basenji', '015.Basset_hound', '016.Beagle', '017.Bearded_collie', '018.Beauceron', '019.Bedlington_terrier', '020.Belgian_malinois', '021.Belgian_sheepdog', '022.Belgian_tervuren', '023.Bernese_mountain_dog', '024.Bichon_frise', '025.Black_and_tan_coonhound', '026.Black_russian_terrier', '027.Bloodhound', '028.Bluetick_coonhound', '029.Border_collie', '030.Border_terrier', '031.Borzoi', '032.Boston_terrier', '033.Bouvier_des_flandres', '034.Boxer', '035.Boykin_spaniel', '036.Briard', '037.Brittany', '038.Brussels_griffon', '039.Bull_terrier', '040.Bulldog', '041.Bullmastiff', '042.Cairn_terrier', '043.Canaan_dog', '044.Cane_corso', '045.Cardigan_welsh_corgi', '046.Cavalier_king_charles_spaniel', '047.Chesapeake_bay_retriever', '048.Chihuahua', '049.Chinese_crested', '050.Chinese_shar-pei', '051.Chow_chow', '052.Clumber_spaniel', '053.Cocker_spaniel', '054.Collie', '055.Curly-coated_retriever', '056.Dachshund', '057.Dalmatian', '058.Dandie_dinmont_terrier', '059.Doberman_pinscher', '060.Dogue_de_bordeaux', '061.English_cocker_spaniel', '062.English_setter', '063.English_springer_spaniel', '064.English_toy_spaniel', '065.Entlebucher_mountain_dog', '066.Field_spaniel', '067.Finnish_spitz', '068.Flat-coated_retriever', '069.French_bulldog', '070.German_pinscher', '071.German_shepherd_dog', '072.German_shorthaired_pointer', '073.German_wirehaired_pointer', '074.Giant_schnauzer', '075.Glen_of_imaal_terrier', '076.Golden_retriever', '077.Gordon_setter', '078.Great_dane', '079.Great_pyrenees', '080.Greater_swiss_mountain_dog', '081.Greyhound', '082.Havanese', '083.Ibizan_hound', '084.Icelandic_sheepdog', '085.Irish_red_and_white_setter', '086.Irish_setter', '087.Irish_terrier', '088.Irish_water_spaniel', '089.Irish_wolfhound', '090.Italian_greyhound', '091.Japanese_chin', '092.Keeshond', '093.Kerry_blue_terrier', '094.Komondor', '095.Kuvasz', '096.Labrador_retriever', '097.Lakeland_terrier', '098.Leonberger', '099.Lhasa_apso', '100.Lowchen', '101.Maltese', '102.Manchester_terrier', '103.Mastiff', '104.Miniature_schnauzer', '105.Neapolitan_mastiff', '106.Newfoundland', '107.Norfolk_terrier', '108.Norwegian_buhund', '109.Norwegian_elkhound', '110.Norwegian_lundehund', '111.Norwich_terrier', '112.Nova_scotia_duck_tolling_retriever', '113.Old_english_sheepdog', '114.Otterhound', '115.Papillon', '116.Parson_russell_terrier', '117.Pekingese', '118.Pembroke_welsh_corgi', '119.Petit_basset_griffon_vendeen', '120.Pharaoh_hound', '121.Plott', '122.Pointer', '123.Pomeranian', '124.Poodle', '125.Portuguese_water_dog', '126.Saint_bernard', '127.Silky_terrier', '128.Smooth_fox_terrier', '129.Tibetan_mastiff', '130.Welsh_springer_spaniel', '131.Wirehaired_pointing_griffon', '132.Xoloitzcuintli', '133.Yorkshire_terrier']
app = Flask(__name__)

def get_model():
    print("init model")
    json_file = open(model_file,'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    print("Loaded Model from disk")
    loaded_model.load_weights(model_weights)

    #compile and evaluate loaded model
    loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    graph = tf.get_default_graph()
    return loaded_model,graph

def predict(tensor):
  prediction = model.predict(np.expand_dims(tensor, axis=0))
  print("prediction")
  print(prediction)
  return prediction

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return x

# load model
model_weights = 'weights.hdf5'
model_file = 'model.json'
model, graph = get_model()


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    file_path = ''
    best_prediction_probability = None
    best_prediction_breed = None
    best_prediction = None
    breed_code = None
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
            image_tensor = path_to_tensor('static/'+file_path)
            with graph.as_default():
                predictions = predict(image_tensor)
                best_prediction = np.argmax(predict(image_tensor), axis=1)[0]
                best_prediction_probability = predictions[0][best_prediction]
                best_prediction_breed = DOG_NAMES[best_prediction][4:]
                if (best_prediction < 10):
                    breed_code = '00'+str(best_prediction)
                elif (best_prediction > 9 and best_prediction < 100):
                    breed_code = '0'+str(best_prediction)
                else:
                    breed_code = str(best_prediction)
                predicted_sample = 'static'

    return render_template('index.html', file_path = file_path, best_prediction_breed = best_prediction_breed, best_prediction_probability = best_prediction_probability, breed_code = breed_code)


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
