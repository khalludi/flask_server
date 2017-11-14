#!flask/bin/python
from flask import Flask, request, jsonify 

# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random
from tf_helpers import *

# restore all of our data structures
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']
context = {}

# import our chat-bot intents file
import json
with open('intents.json') as json_data:
  intents = json.load(json_data)

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# load our saved model
model.load('./model.tflearn')

app = Flask(__name__)

@app.route('/tf_model/predict', methods=['POST'])
def tf_predict():
    global context
    if not request.json or not 'message' in request.json or not 'user-id' in request.json:
        abort(400)
    if request.content_type != 'application/json':
        abort(400)
    try:
    	new_msg = json.loads(request.data)
    except ValueError:
        abort(400)
    
    res = response(request.json['message'], model, intents, words, request.json['user-id'], False, context)
    ret = {
        'message': res['message']
    }
    context = res['context']
    
    return jsonify(ret)

if __name__ == '__main__':
  app.run(debug=True)
