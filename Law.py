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
import glob
import errno

import nltk
import re
import random
import numpy as np
from nltk.book import *

stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


path = '/Users/AlexWang/Law case-200/*.txt'
files = glob.glob(path)

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

a_words = []
r_words = []
for i in doc_list:
    if i[1] == 1:
        a_words.append(i[0])
    elif i[1] == 0:
        r_words.append(i[0])
a_words = [item for items in a_words for item in items]
r_words = [item for items in r_words for item in items]         
print(a_words)
print(r_words)
freq_a = FreqDist(a_words)
freq_r = FreqDist(r_words)
sorted_freq_a = sorted(freq_a.items(),key = lambda k:k[1], reverse = True)
sorted_freq_r = sorted(freq_r.items(),key = lambda k:k[1], reverse = True)

freq_a.plot(30)
freq_r.plot(30)
#%%
print(freq_a.most_common(50))
print(freq_r.most_common(50))
#%%
import pandas as pd
from sklearn import feature_extraction
totalvocab_stemmed = []
totalvocab_tokenized = []
def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word not in stopwords]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


for i in doc:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)


from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,1))

tfidf_matrix = tfidf_vectorizer.fit_transform(doc) #fit the vectorizer to synopses


terms = tfidf_vectorizer.get_feature_names()
from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()
from sklearn.externals import joblib



joblib.dump(km,  'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :10]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace
