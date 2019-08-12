#!/usr/bin/python3

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
#Download dependencies
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from flask_cors import CORS, cross_origin

import pickle
import pandas as pd
import numpy as np
import json
import os
import random


from flask import Flask, request, jsonify

from werkzeug.utils import secure_filename

from lsa.preprocessor import spellcorrect
from lsa.CHATBOT_LSI import lsa
from lsa.SCRAPER_LSI import scraper_lsa

from inference import load_model
os.system('scrapy crawl faqspider')
file = open("data.txt", "r") 
unstructured_data = file.read() 



import tensorflow as tf
data = pickle.load( open( "seq2seq/keras-assistant-data.pkl", "rb" ) )
words = data['words']
classes = data['classes']

with open('seq2seq/intents.json') as json_data:
    intents = json.load(json_data)

fallback_dict = ["I'm sorry, I don't know that yet", "Can you please rephrase that?", "I do not have any info on that yet.", "I am still learning, can you type that out in a different way?"]


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    # print ("found in bag: %s" % w)
                    continue

    return(np.array(bag))


# p = bow("Hello", words)
# print (p)
# print (classes)



# Use pickle to load in the pre-trained model
global graph
graph = tf.get_default_graph()

with open('seq2seq/keras-assistant-model.pkl', 'rb') as f:
    seq2seq_model = pickle.load(f)

def classify_local(sentence):
    ERROR_THRESHOLD = 0.50
    
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = seq2seq_model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    # return tuple of intent and probability
    print("Seq2Seq Result : ", return_list)
    return return_list


classify_local('Hello World!')

def small_talk(input_text):
      # global stress
    ERROR_THRESHOLD = 0.9999
    context = None
    sentence = request.json['query']
    # print(sentence)
    # sentiment = sentiment_analysis.sentiment_analyzer(sentence)
    
    if "context" in request.json:
        print("Context present : ", request.json['context'])
        context = request.json['context']

    else:
        print("Context not present!")
        context = None

    # entities = entity_extraction.named_entity_extraction(sentence)
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = seq2seq_model.predict([input_data])[0]
    # filter out predictions below a threshold

    print("Before Filter : ", results)
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]

    print("After Filter : ", results)
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    output_context = None
    print("Results Length : ", len(results))
    if len(results) == 0:
        print("Fallback detected inside seq2seq")
        return "fallback"
        # return_list.append({"query": sentence, "intent": "fallback", "response": random.choice(fallback_dict), "context": None, "probability": "0.00"})
    else:
        print("Inference Exists")
        for r in results:
            if context != None:
                    classes[r[0]] = context
                    print("Class Value : ", classes[r[0]])

            for x_tend in intents['intents']:
                
                if classes[r[0]] == x_tend['tag']:
                    # print("Entities Length : ", len(entities))
                    if x_tend['context'] == "":
                        output_context = None
 
                    return random.choice(x_tend['responses'])


def get_response(input_text):
    print("Input Text : {}".format(input_text))
   
    # spellcorrected_text = spellcorrect(input_text)

    print("Calling Small Talk Keras Model")

    seq2seq_response = small_talk(input_text)
    print("Small Talk Response : {}".format(seq2seq_response))

    if seq2seq_response == "fallback":
        #call lsa
        lsa_response = lsa(input_text)
        print("LSA response : {}".format(lsa_response))

        if lsa_response == "fallback":
            #call crawler
            

            scraper_lsa_response = scraper_lsa(input_text)
            if scraper_lsa_response == "fallback":
                #Call Glove and SQUAD
                print("Scraper LSA fallback!")
                print("Call GloVe and Squad")
           
                query = input_text
                answer = ' '.join(model.ask(unstructured_data, query))
                return {"glove": answer}
            else:
                return scraper_lsa_response
        else:
            return lsa_response
    else:
        return seq2seq_response


    # file_path = "/path/to/yourfile.txt"
    file_path = os.getcwd() + '/fallback_sentences.txt'
    with open(file_path, 'a') as file:
        file.write(input_text + "\n")

app = Flask(__name__)
model = load_model()
print("model loaded")
CORS(app)


@app.route("/", methods=['GET'])
def health():
    return jsonify({
        "status":"active",
        "state":"running",
        "errors":None
    })

#Example for Root Route
@app.route('/query', methods=['POST'])
def lsa_processor():
    # print(request.get_json())
    if not request.json:
        return jsonify({
            "status": 400,
            "message": "Bar request. Request has no body"
        })
    else:
        print(request.get_json())
        print(type(get_response(request.get_json()['query'])))
        payload = get_response(request.get_json()['query'])
        # print("Respone sending out", payload)
        print(type(payload))
        if (type(payload) == str):
            return jsonify({
                "result" : {
                    "fulfillment":{
                        "messages": [{
                            "type": 0,
                            "platform": "facebook",
                            "speech": payload
                            }
                        ]
                    }        
                }
            })
        else:
            response_text = payload['glove'] + '. Did this resolve your query?'
            return jsonify({ "result" : {

                            "fulfillment":{
                                "speech": "",
                                "displayText": "",
                                "messages": [{
                                    "type": 4,
                                    "platform": "facebook",
                                    "payload": {
                                        "facebook": {
                                            "text": response_text,
                                            "quick_replies": [{
                                                    "content_type": "text",
                                                    "title": 'Yes',
                                                    "payload": 'satisfactoryworkflowcontinuity'
                                                }, {
                                                    "content_type": "text",
                                                    "title": 'No',
                                                    "payload": 'discardedchannelagent'
                                                }]
                                        }
                                    }
                                }]
                            }
                        }
                    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

    