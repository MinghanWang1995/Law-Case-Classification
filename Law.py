# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 17:23:55 2018

@author: AlexWang
"""
#%%

import glob
import errno

import nltk
import re
import random
import numpy as np


stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


path = '/Users/AlexWang/Law case-200/*.txt'
files = glob.glob(path)



cate = []
category = [0,1,2,3,4,5,6,7,8,9,10]
for name in files:
    try:
        with open (name, 'r', encoding = 'latin-1') as f:
            #print(name)
           # cat = 0
            for line in f:
            #print(line)
                line = line.lower()
                if line == "affirmed.\n":
                    cat = 1
                elif line.startswith("affirmed in part, reversed in part, and remanded."):
                    cat = 2
   
                elif line.startswith("reversed and remanded"):
                    cat = 3

                elif line == "reversed.\n":
                    cat = 0
                     
                elif line.startswith("affirmed in part, reversed in part, and vacated"):
                    cat = 4
                   
                elif line.startswith("vacated and remanded"):
                    cat = 5
                    
                elif line.startswith("affirmed in part and reversed in part"):
                    cat = 6
                   
                elif line.startswith("appeal dismissed; affirmed"):
                    cat = 7
                    
                elif line.startswith("affirmed in part, vacated in part, and remanded"):
                    cat = 8
                    
                elif line.startswith("affirmed in part and modified in part"):
                    cat = 9
                  
                elif line.startswith("motion denied"):
                    cat = 10
            cate.append(cat)
            
            
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
            
print("Finished categorizing.")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word not in stopwords]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


doc = []
for name in files:
    try:
        with open (name, 'r', encoding = 'utf-8') as f:
            doc.append(f.read().replace('\n', ''))
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise  

doc_list = []
for i in range(len(doc)):
    words = tokenize_and_stem(doc[i])
    doc_list.append((words,cate[i]))
    
#print(doc_list[0])
 
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(category)
 
 
for doc in doc_list:
    # initialize our bag of words(bow) for each document in the list
    bow = []
    # list of tokenized words for the pattern
    token_words = doc[0]
    # stem each word
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    # create our bag of words array
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)
     
    output_row = list(output_empty)
    output_row[category.index(doc[1])] = 1
     
    # our training set will contain a the bag of words model and the output row that tells which catefory that bow belongs to.
    training.append([bow, output_row])
 
# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(training)
training = np.array(training)
 
# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:150,0])
train_y = list(training[:150,1])   
test_x = list(training[150:,0])
test_y = list(training[150:,1])

print("Finished preprocessing.")





        
#%%
import tensorflow as tf
import tflearn
tf.reset_default_graph()
print("Start training.")
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
 
# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

with tf.Session() as sess:
    # Start training (apply gradient descent algorithm)
    model.fit(train_x, train_y, n_epoch=20000, batch_size=8, show_metric=True)
    pred_y = model.predict(test_x)
    correct_prediction = tf.equal(tf.arg_max(pred_y, 1), tf.arg_max(test_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("acciracy: ", sess.run(accuracy))
    

#%%