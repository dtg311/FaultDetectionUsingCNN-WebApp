import sys
import numpy as np
import keras as k
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__,template_folder='templates')
from keras.models import load_model
from keras.backend import set_session
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
#msg = tf.constant('Hello, TensorFlow!')
#tf.disable_v1_behavior()
print("Loading model")
#global sess
#sess = tf.compat.v1.Session()
#set_session(sess)
#global model
model = load_model('mlmodel.h5')
#global graph
#graph = tf.get_default_graph()

#import pdb; pdb.set_trace()
@app.route('/', methods=['GET', 'POST']) 
def main_page():
    if request.method == 'POST':

        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
        
    return render_template('index.html')

@app.route('/prediction/<filename>')
def prediction(filename):
    #Step 1
    my_image = plt.imread(os.path.join('uploads', filename))
    #Step 2
    im=my_image
    im1 = im.reshape(1,40, 24, 1)
    im1 = im1 / 255
    
    #Step 3
    #with graph.as_default():
    #set_session(sess)
    probabilities = model.predict(im1)
    print('probabilities:',probabilities)
#Step 4
    predictions= {0:'cell', 1:'cell multi', 2:'cracking',3:'diode',4:'diode multi',5:'hotspot',6:'hotspot multi',7:'no anomaly',8:'offline module',9:'shadowing',10:'soiling', 11:'vegetation'}
    index = np.argmax(probabilities)
    print('index:',index)
    
#Step 5
    return render_template('predict.html', predictions=predictions,index=index)

app.run(host='0.0.0.0', port=80)
