from flask import Flask, render_template, request
from keras.preprocessing import image
from keras.models import load_model
from keras.utils import img_to_array
from keras.utils import load_img
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import os
import sys

# Loading model
model = load_model("F:/Quark/ML workshop/ML workshop project/vegetables.h5")

class_map = {0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: 'Broccoli', 5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'}

# Preparing and pre-processing the image
def preprocess_predict_img(img_path):  
    print(1)
    print(img_path)
    test_img = image.load_img(img_path, target_size=(150, 150))
    print(2)
    test_img_arr = image.img_to_array(test_img)/255.0
    print(3)
    test_img_input = test_img_arr.reshape((1, test_img_arr.shape[0], test_img_arr.shape[1], test_img_arr.shape[2]))
    print(4)
    predicted_label = np.argmax(model.predict(test_img_input))
    print(5)
    predicted_vegetable = class_map[predicted_label]
    print(6)
    return predicted_vegetable


# Instantiating flask app
app = Flask(__name__)

@app.route("/")
def main():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def predict_image_file():
    try:
        if request.method == 'POST':
            img = request.files['file']
            # print(img, file=sys.stderr)            
            basepath = os.path.dirname(os.path.abspath(__file__))
            # print(basepath, file=sys.stderr)            
            img_path = os.path.join(basepath, 'uploads', secure_filename(img.filename))
            img.save(img_path)
            # print(img_path, file=sys.stderr)            
            print("Above Prediction", file=sys.stderr)
            pred = preprocess_predict_img(img_path)
            print("Prediction: "+pred, file=sys.stderr)
            return render_template("result.html", predictions=pred, img_path = img_path)

    except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run()