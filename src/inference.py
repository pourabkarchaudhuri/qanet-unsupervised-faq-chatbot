import os
import tensorflow as tf
import numpy as np
from keras import backend as K
# import sys 
# sys.path.append('..')
from config import Config as cf
from common import SquadData, Span, Answer
from model import QANetModel

from utilities import augment_long_text, tokenize, tokenize_long_text, to_chars, align

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# tf_config = tf.ConfigProto(allow_soft_placement=True)
# tf_config.gpu_options.allow_growth = True
# K.set_session(tf.Session(config=tf_config))

def load_model():
    print("load_model called")
    
    data = SquadData.load(cf.SQUAD_DATA)
    data_binary = SquadData.load(np_path=cf.SQUAD_NP_DATA)
    word_vectors, char_vectors, train_ques_ids, X_train, y_train, val_ques_ids, X_valid, y_valid = data_binary
    model = QANetModel(cf, data_binary, data)
    filename = cf.INFERENCE_MODEL_PATH
    model.load_weights(filename)

    # Predict once to overcome the error in API request
    context = "In early 2012, NFL Commissioner Roger Goodell stated that the league planned to make the 50th Super Bowl \"spectacular\" and that it would be \"an important game for us as a league\"."
    query = "Which Super Bowl did Roger Goodell speak about?"
    answer = model.ask(context, query)
    print(answer)

    return model

##############
# Start Flask:
# $ FLASK_APP=demo_qanet.py flask run --host=0.0.0.0 --port=8080
