# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 03:00:54 2021

@author: Mahsa
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 19:47:44 2021

@author: 
"""

import nltk
import gensim
import numpy as np
import math
from nltk.tokenize import word_tokenize , sent_tokenize
data = "Mars is approximately half the diameter of Earth."
# print(word_tokenize(data))
data = "Mars is a cold desert world. It is half the size of Earth. "
# print(sent_tokenize(data))
file_docs = []
nltk.download('punkt')

with open ('demofile.txt') as f1:
    tokens = sent_tokenize(f1.read())
    for line in tokens:
        file_docs.append(line)
# print("Number of documents:",len(file_docs))
gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in file_docs]
#print(doc2bag)
dictionary = gensim.corpora.Dictionary(gen_docs)
# print(dictionary.token2id)
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
# print(corpus)
tf_idf = gensim.models.TfidfModel(corpus)
#TF_IDF
for doc in tf_idf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])
sims = gensim.similarities.Similarity('newfolder',tf_idf[corpus],
                                        num_features=len(dictionary))
print(sims)

#TF
def TF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    
    for word, count in wordDict.items():
        tfDict[word] =count/ float(bagOfWordsCount)
    return tfDict
tf=TF(numOfWords, bagOfWords)
#IDF
def IDF(documents):
    import math
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val >0:
                idfDict[word] += 1   
    for word, val in idfDict.items():
        idfDict[word] = math.log(N/float(val)) 
    
    return idfDict
#SIM
def cosinesim(dic1,dic2):
    numerator = 0
    dena = 0
    for key1,val1 in dic1.items():
        numerator += val1*dic2.get(key1,0.0)
        dena += val1*val1
    denb = 0
    for val2 in dic2.values():
        denb += val2*val2
    sim= numerator/math.sqrt(dena*denb)
    return sim
 
file2_docs = []
with open ('demofile2.txt') as f2:
    tokens = sent_tokenize(f2.read())
    for line in tokens:
        file2_docs.append(line)
print("Number of documents:",len(file2_docs))  
for line in file2_docs:
    query_doc = [w.lower() for w in word_tokenize(line)]
    query_doc_bow = dictionary.doc2bow(query_doc)
query_doc_tf_idf = tf_idf[query_doc_bow]

with open('demofile.txt') as document:
    bagOfWords=[word for line in document for word in line.split()]
with open('demofile2.txt') as document2:
    bagOfWords2=[word for line in document2 for word in line.split()]

uniqueWords = set(bagOfWords).union(set(bagOfWords2))
numOfWords = dict.fromkeys(uniqueWords,0)
numOfWords2 = dict.fromkeys(uniqueWords,0)
for word in bagOfWords: 
    numOfWords[word] +=1
tf=TF(numOfWords,bagOfWords) 
print('tf: ', tf) 
for word in bagOfWords2: 
    numOfWords2[word] +=1
query_doc_tf=TF(numOfWords2,bagOfWords2) 
idfs = IDF([numOfWords, numOfWords2]) 
idf={}
for x,y in idfs.items():
    if x in bagOfWords:
        idf[x]=y
        
idf2={}
for x,y in idfs.items():
    if x in bagOfWords2:
        idf2[x]=y

print('idf: ', idf)
print('query_doc_tf: ', query_doc_tf)
print('query_doc_idf: ', idf2)
# print(document_number, document_similarity)
print('Comparing Result By TF_IDF:', sims[query_doc_tf_idf]) 
print('Comparing Result By TF:', cosinesim(tf,query_doc_tf)) 
print('Comparing Result By IDF:', cosinesim(idf, idf2))
sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))
print(sum_of_sims)
percentage_of_similarity = round(float((sum_of_sims / len(file_docs)) * 100))
print(f'Average similarity float: {float(sum_of_sims / len(file_docs))}')
print(f'Average similarity percentage: {float(sum_of_sims / len(file_docs)) * 100}')
print(f'Average similarity rounded percentage: {percentage_of_similarity}')
avg_sims = [] # array of averages

# for line in query documents
for line in file2_docs:
        # tokenize words
        query_doc = [w.lower() for w in word_tokenize(line)]
        # create bag of words
        query_doc_bow = dictionary.doc2bow(query_doc)
        # find similarity for each document
        query_doc_tf_idf = tf_idf[query_doc_bow]
        # print (document_number, document_similarity)
        print('Comparing Result:', sims[query_doc_tf_idf]) 
        # calculate sum of similarities for each query doc
        sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))
        # calculate average of similarity for each query doc
        avg = sum_of_sims / len(file_docs)
        # print average of similarity for each query doc
        print(f'avg: {sum_of_sims / len(file_docs)}')
        # add average values into array
        avg_sims.append(avg)  
    # calculate total average
total_avg = np.sum(avg_sims, dtype=np.float)
    # round the value and multiply by 100 to format it as percentage
percentage_of_similarity = round(float(total_avg) * 100)
    # if percentage is greater than 100
    # that means documents are almost same
if percentage_of_similarity >= 100:
    percentage_of_similarity = 100
    

