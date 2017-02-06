import os, os.path
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import math

## Split headers from text ## 
def data_splitter(myFile):
    ff = open(myFile)
    content = ff.read()    
    myclass = ""
    ## Split data based on the first occurence of an empty line ##
    data = content.split("\n\n", 1)
    classdata = data[0].split('\n')
    for var in classdata:
        if("Newsgroups" in var):
            blah = var.split(": ")
            myclass = blah[1].strip()
    
    return data[1], myclass

## tokenize string ##
def tokenizeString(mystring):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    return tokenizer.tokenize(mystring)

## Count words ##
def createVector(words):
    vector = dict.fromkeys(words,0)
    for token in words:
        vector[token] += 1
    return vector

## Common method ##
def getVectorTokens(content):
    tokenizedContent = tokenizeString(content.lower())
    myvector = createVector(tokenizedContent)
    return myvector

## Get maximum argument from all probabilities ##
def getArgMax(priorProbs, allProbs, fileVector):
    finalProb = {}
    tempVar = 0
    ## calculate probability of each class ##
    for key in allProbs:
        tempVar = priorProbs[key]
        for key2 in fileVector:
            blah=allProbs[key]
            if key2 in blah:
                tempVar = tempVar * math.pow(blah[key2],fileVector[key2])
        
        finalProb[key] = tempVar
    
    ## get the class with maximum probability ##
    maxVal = 0
    maxClass = ''
    for mykey in finalProb:
        if maxVal < finalProb[key]:
            maxVal = finalProb[mykey]
            maxClass = mykey
    
    return maxClass


DIR = './data/20_newsgroups'
b = os.listdir(DIR)
folderList = {}

## Save all the files in a datastructure so that half can be used for training and the other half for testing ##
print('Reading files from the data folder')
for foldername in b:
    fileList = []
    fileList.extend([name for name in os.listdir(DIR+"/"+foldername) 
        if os.path.isfile(os.path.join(DIR+"/"+foldername, name))])
    folderList[foldername] = fileList        

classwisecontent = {}
allFileVector = {}
n = {}

## Get frequency of all the words in the document ##
for key in folderList:
    vals = folderList[key]
    vals.sort()        
    for i in range(0, int((len(vals))/2)):
        aFileVector = {}
        mycontent, myclassname = data_splitter(DIR+"/"+key+"/"+vals[i])
        aFileVector = getVectorTokens(mycontent)
        allFileVector = {k: aFileVector.get(k, 0) + allFileVector.get(k, 0) for k in set(aFileVector) | set(allFileVector)}

    classwisecontent[key] = allFileVector

## Get total number of words in a class ##
for key in classwisecontent:
    total = 0
    mytempVals = classwisecontent[key]
    for word in mytempVals:
        total = total + mytempVals[word]
    
    n[key] = total

## Get total number of distinct words in all the files ##
vocab = {}
for key in classwisecontent:
    vals = classwisecontent[key]     
    vocab = {k: vals.get(k, 0) + vocab.get(k, 0) for k in set(vals) | set(vocab)}

len_vocab = len(vocab)

## Training data ##
print('Training data is being processed to calculate word probabilities')
finalProbabilities = {}
classWiseProbabilities = {}
for key in classwisecontent:
    tempDict=classwisecontent[key]
    for wordList in tempDict:
        value = tempDict[wordList]
        ## for each word probability is its frequency in the document divided by 
        ## total number of words in the documents (for the class) and total number of distinct words in all documents
        prob_word_given_class = ((value+1)/(n[key]+len_vocab))
        classWiseProbabilities[wordList] = prob_word_given_class

    finalProbabilities[key] = classWiseProbabilities

## Get total documents for prior probabilities ##
total_docs = 0
for key in folderList:
    vals = folderList[key]
    total_docs = total_docs + len(vals)

## Calculate prior probabilities ##
print('Calculating prior probabilities')
prior_probabilities = {}
for key in folderList:
    vals = folderList[key]
    class_len = len(vals)
    prior_probabilities[key] = class_len/total_docs

## Testing data ##
print('Classifying testing data')
correct = 0
incorrect = 0
for key in folderList:
    vals = folderList[key]
    vals.sort()        
    for i in range((int((len(vals))/2)+1), len(vals)):
        aFileVector = {}
        ## removes headers and newsgroup name for obtaining accuracy ##
        mycontent, myclassname = data_splitter(DIR+"/"+key+"/"+vals[i])
        aFileVector = getVectorTokens(mycontent)
        ## Get predicted class by selecting the maximum probability for a class ##
        predicted_class = getArgMax(prior_probabilities, finalProbabilities, aFileVector)
        ## if predicted class is present in class name the classification is correct ##
        if predicted_class in myclassname:
            correct = correct + 1
        else:
            incorrect = incorrect + 1
        
accuracy = ((correct/(correct + incorrect))*100)
print ('The accuracy is: '+ str(accuracy))
