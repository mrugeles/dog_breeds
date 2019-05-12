import numpy as np
import cv2
import os

import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.xception import Xception, preprocess_input
from prediction import Prediction


class ModelUtilities:
    DOG_NAMES =['001.Affenpinscher', '002.Afghan_hound', '003.Airedale_terrier', '004.Akita', '005.Alaskan_malamute', '006.American_eskimo_dog', '007.American_foxhound', '008.American_staffordshire_terrier', '009.American_water_spaniel', '010.Anatolian_shepherd_dog', '011.Australian_cattle_dog', '012.Australian_shepherd', '013.Australian_terrier', '014.Basenji', '015.Basset_hound', '016.Beagle', '017.Bearded_collie', '018.Beauceron', '019.Bedlington_terrier', '020.Belgian_malinois', '021.Belgian_sheepdog', '022.Belgian_tervuren', '023.Bernese_mountain_dog', '024.Bichon_frise', '025.Black_and_tan_coonhound', '026.Black_russian_terrier', '027.Bloodhound', '028.Bluetick_coonhound', '029.Border_collie', '030.Border_terrier', '031.Borzoi', '032.Boston_terrier', '033.Bouvier_des_flandres', '034.Boxer', '035.Boykin_spaniel', '036.Briard', '037.Brittany', '038.Brussels_griffon', '039.Bull_terrier', '040.Bulldog', '041.Bullmastiff', '042.Cairn_terrier', '043.Canaan_dog', '044.Cane_corso', '045.Cardigan_welsh_corgi', '046.Cavalier_king_charles_spaniel', '047.Chesapeake_bay_retriever', '048.Chihuahua', '049.Chinese_crested', '050.Chinese_shar-pei', '051.Chow_chow', '052.Clumber_spaniel', '053.Cocker_spaniel', '054.Collie', '055.Curly-coated_retriever', '056.Dachshund', '057.Dalmatian', '058.Dandie_dinmont_terrier', '059.Doberman_pinscher', '060.Dogue_de_bordeaux', '061.English_cocker_spaniel', '062.English_setter', '063.English_springer_spaniel', '064.English_toy_spaniel', '065.Entlebucher_mountain_dog', '066.Field_spaniel', '067.Finnish_spitz', '068.Flat-coated_retriever', '069.French_bulldog', '070.German_pinscher', '071.German_shepherd_dog', '072.German_shorthaired_pointer', '073.German_wirehaired_pointer', '074.Giant_schnauzer', '075.Glen_of_imaal_terrier', '076.Golden_retriever', '077.Gordon_setter', '078.Great_dane', '079.Great_pyrenees', '080.Greater_swiss_mountain_dog', '081.Greyhound', '082.Havanese', '083.Ibizan_hound', '084.Icelandic_sheepdog', '085.Irish_red_and_white_setter', '086.Irish_setter', '087.Irish_terrier', '088.Irish_water_spaniel', '089.Irish_wolfhound', '090.Italian_greyhound', '091.Japanese_chin', '092.Keeshond', '093.Kerry_blue_terrier', '094.Komondor', '095.Kuvasz', '096.Labrador_retriever', '097.Lakeland_terrier', '098.Leonberger', '099.Lhasa_apso', '100.Lowchen', '101.Maltese', '102.Manchester_terrier', '103.Mastiff', '104.Miniature_schnauzer', '105.Neapolitan_mastiff', '106.Newfoundland', '107.Norfolk_terrier', '108.Norwegian_buhund', '109.Norwegian_elkhound', '110.Norwegian_lundehund', '111.Norwich_terrier', '112.Nova_scotia_duck_tolling_retriever', '113.Old_english_sheepdog', '114.Otterhound', '115.Papillon', '116.Parson_russell_terrier', '117.Pekingese', '118.Pembroke_welsh_corgi', '119.Petit_basset_griffon_vendeen', '120.Pharaoh_hound', '121.Plott', '122.Pointer', '123.Pomeranian', '124.Poodle', '125.Portuguese_water_dog', '126.Saint_bernard', '127.Silky_terrier', '128.Smooth_fox_terrier', '129.Tibetan_mastiff', '130.Welsh_springer_spaniel', '131.Wirehaired_pointing_griffon', '132.Xoloitzcuintli', '133.Yorkshire_terrier']



    def __init__(self, model_path):
        self.model, self.graph = self.get_model(model_path)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.face_cascade = cv2.CascadeClassifier(dir_path + '/haarcascades/haarcascade_frontalface_alt.xml')

    def get_model(self, model_path):
        graph = tf.get_default_graph()
        return load_model(model_path), graph

    def extract_Xception(sefl, tensor):
    	return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

    def path_to_tensor(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        return np.expand_dims(x, axis=0)

    def predict_breed(self, img_path):
        breed = ""
        breed_index = -1
        bottleneck_feature = bottleneck_feature = self.extract_Xception(self.path_to_tensor(img_path))
        predicted_vector = self.model.predict(bottleneck_feature)

        probability = np.amax(predicted_vector)
        breed_index = np.argmax(predicted_vector) + 1

        breed = self.DOG_NAMES[breed_index][4:]
        is_dog = False if(probability < 0.5) else True
        return breed, breed_index, is_dog

    def face_detector(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def predict_image(self, img_path):
        breed, breed_code, is_dog = self.predict_breed(img_path)
        print("Prediction result: %s, %d"%(breed, breed_code))
        if(is_dog):
            return Prediction.DOG, breed, breed_code
        elif(self.face_detector(img_path)):
            return Prediction.PERSON, breed, breed_code
        else:
            return Prediction.UNKNOWN, breed, breed_code
