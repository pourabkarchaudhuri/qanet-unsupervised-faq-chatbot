import pandas as pd
import os
import random
import nltk
import string
from nltk.corpus import stopwords 
import gensim
from gensim import corpora, models, similarities

import warnings
warnings.simplefilter('ignore')


# Data Preparation and Preprocessing

print("Loading Data Sources")
#Load Data Source
data_path = os.getcwd() + '\\lsa\\dataset.json'
print("Data source : " + data_path)
data = pd.read_json(data_path)
data.head()

temp_data = pd.read_json(data_path)
question_data = temp_data['MESSAGE']
#Create Stop Word
newstopwords = set(stopwords.words('english'))
#define Wordnet Lemmatizer 
WNlemma = nltk.WordNetLemmatizer()

#Create Preprocessing Function
def pre_process(text):
    tokens = nltk.word_tokenize(text)
    tokens=[WNlemma.lemmatize(t) for t in tokens]
    tokens= [ t for t in tokens if t not in string.punctuation ]
    tokens=[word for word in tokens if word.lower() not in newstopwords]
    # bigr = nltk.bigrams(tokens[:10])
    # trigr = nltk.trigrams(tokens[:10])
    return(tokens)

#greeting function
# GREETING_INPUTS = ("hello", "hi", "greetings", "hello i need help", "good day","hey","i need help", "greetings")
# GREETING_RESPONSES = ["Good day, How may i of help?", "Hello, How can i help?", "hello", "I am glad! You are talking to me."]
           
# def greeting(sentence):
#     for word in sentence.split():
#         if word.lower() in GREETING_INPUTS:
#             return random.choice(GREETING_RESPONSES)



#Preprocess Question Column
# question_data = data['MESSAGE']
data['MESSAGE'] = data['MESSAGE'].apply(pre_process)
    
#Define Questions
question = data['MESSAGE']


dictionary = corpora.Dictionary(question)
corpus = [dictionary.doc2bow(a) for a in question]
tfidf = models.TfidfModel(corpus)
    
corpus_tfidf = tfidf[corpus]
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=650) # Threshold A
corpus_lsi = lsi[corpus_tfidf]
index = similarities.MatrixSimilarity(corpus_lsi)


# ChatBot Function Definition


def Talk_To_Tau(test_set_sentence):      
    print("Test sentence : {}".format(test_set_sentence))  
    # ---------------Tokenisation of user input -----------------------------#
    tokens = pre_process(test_set_sentence)
    texts = " ".join(tokens)    
    # -----------------------------------------------------------------------#
    
    # ---------------Find and Sort Similarity -------------------------------#
    vec_bow = dictionary.doc2bow(texts.lower().split())
    vec_tfidf = tfidf[vec_bow]
    vec_lsi = lsi[vec_tfidf]

    #If not in the topic trained.
    if not (vec_lsi):
        
        not_understood = "fallback"
        return not_understood, 999
    
    else: 
        # sort similarity
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        
        index_s =[]
        score_s = []
        for i in range(len(sims)):
            x = sims[i][1]
            print("Threshold : {}".format(x))
            # If similarity is less than 0.5 ask user to rephrase.
            if x <=0.8: # Threshold B
                print("Below Threshold.")
                index_s.append(str(sims[i][0]))
                score_s.append(str(sims[i][1]))
                reply_indexes = pd.DataFrame({'index': index_s,'score': score_s})

                r_index = int(reply_indexes['index'].loc[0])
                r_score = float(reply_indexes['score'].loc[0])

                # not_understood = "Apology, I do not understand. Can you rephrase?"
                # return not_understood, 999
                if (r_index < len(question_data) - 3):
                    return "fallback", 999
                else:
                    not_understood = "fallback"
                    return not_understood, 999
            else: 
                index_s.append(str(sims[i][0]))
                score_s.append(str(sims[i][1]))
                reply_indexes = pd.DataFrame({'index': index_s,'score': score_s})
        

            #Find Top Questions and Score  
            r_index = int(reply_indexes['index'].loc[0])
            r_score = float(reply_indexes['score'].loc[0])
            reply = str(data.iloc[:,1][r_index])
            print("Reply : {}, Score : {}".format(reply, r_score))
            return reply, r_score


def lsa(sentence): 
    print("Input to LSA recieved : {}".format(sentence))
    # sentence = input("User says > ")
    # if(sentence.lower()!='bye'):
    #     if(greeting(sentence.lower())!=None):
    #         # print('Bot says > '+ greeting(sentence.lower()))
    #         return greeting(sentence.lower())
    #     else:
    reply =[]
    score =[]
    reply, score = Talk_To_Tau(str(sentence))
    print("Reply : {}, Score : {}".format(reply, score))
    # print("Score from AlICE : ", score)
    return reply
            
            #For Tracing, comment to remove from print
            #print("")
            #print("SCORE: "+str(score))

