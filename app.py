import os
from uuid import uuid4
import pickle
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")

key_list = pickle.load(open('key_list', 'rb'))
val_list = pickle.load(open('val_list', 'rb'))
key_list_sub = pickle.load(open('key_list_sub', 'rb'))
val_list_sub = pickle.load(open('val_list_sub', 'rb'))


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images1/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        #import tensorflow as tf
        import numpy as np
        from keras.preprocessing import image

        from keras.models import load_model
        new_model = load_model('model.h5')
        new_model_cat = load_model('mysubnewmodel.h5')

        new_model.summary()
        test_image = image.load_img('images1\\'+filename,target_size=(60,80))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = new_model.predict(test_image)
        result1 = new_model_cat.predict(test_image)
        ans = np.argmax(result)
        ans2 = np.argmax(result1)
        ans = key_list[val_list.index(ans)]
        ans2 = key_list_sub[val_list_sub.index(ans2)]
            
    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("template.html",image_name=filename, text=ans , text2=ans2)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images1", filename)

if __name__ == "__main__":
    app.run(debug=False)

