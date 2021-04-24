
import json
import math
import operator
import random
import re
import string
import nltk
import time
import os, sys
from sklearn.model_selection import train_test_split
from nltk import WordNetLemmatizer
from statistics import mode, StatisticsError


def tokenization(file, filenum, StopWords):
    data = []
    tf = {}
    f = open("../content/bbcsport/" + file + "/" + filenum, 'r', encoding='latin-1')
    for line in f:
        line = line.lower()
        line = StopWordsRemoval(line, StopWords)
        for word in line.split():
            data.append(word)
            word = Normalization(word)
            word = Lemmatization(word)
            tf[word] = data.count(word)
    f.close()
    return tf

def Normalization(doc):
    # doc = re.sub(r"(?<!\w)([A-Za-z])\.", r"\1", doc)
    doc = re.sub("^\d+\s|\s\d+\s|\s\d+$", "", doc)
    Reg_ex = re.compile('[%s]' % re.escape(string.punctuation))
    doc = Reg_ex.sub(' ', doc)
    doc = doc.replace('”', " ")
    return doc

def Lemmatization(doc):
    Lemmatizer = WordNetLemmatizer()
    doc = Lemmatizer.lemmatize(doc)
    return doc

def StopWordsRemoval(doc, StopWords):
    doc = doc.split(" ")
    for word in StopWords:
        if word in doc:
            doc.remove(word)
    doc = " ".join(doc)
    doc.replace("\t", ' ')
    doc.replace("    ", ' ')
    doc.replace("   ", ' ')
    doc.replace("  ", ' ')
    return doc

def BagOfWords(data_train, data_dic):
    BOW = []
    outfile = open("../content/BOW.json", "w")
    for key in data_train:
        for word in data_dic[key]['terms'].keys():
            if word not in BOW:
                if (word.isnumeric() == False):
                    BOW.append(word)
    json_object = json.dumps(BOW, indent=0)
    outfile.write(json_object + "\n")
    outfile.close()
    return BOW

def TermFrequency(data_train, data_dic, BOW):
    tf = {}
    for key in data_train:
        doc_id = data_dic[key]['sport'] + ": " + data_dic[key]['id']
        tf[doc_id] = [0] * len(BOW)

        for j in range(0, len(BOW)):
                if BOW[j] in data_dic[key]['terms']:
                    tf[doc_id][j] = data_dic[key]['terms'][BOW[j]]
    return tf

def TF_IDF(data_train, tf, BOW):
    IDF = [0] * len(BOW)
    for i in range(0, len(BOW)):
        df = 0
        for doc_id in tf.keys():
            if tf.get(doc_id)[i] != 0:
                df += 1
            else:
                df = 1
        idf = float(format(math.log10(len(data_train) / df), '.5f'))
        for doc_id in tf.keys():
            tf.get(doc_id)[i] *= idf
    return tf

def ProductOfVectors(doc1, doc2):
    return sum(map(operator.mul, doc1, doc2))

def CosineSimilarity(training_tf, testing_tf):
    prod = ProductOfVectors(training_tf, testing_tf)
    len1 = math.sqrt(ProductOfVectors(training_tf, training_tf))
    len2 = math.sqrt(ProductOfVectors(testing_tf, testing_tf))
    Cosine_Similarity = prod / (len1 * len2)
    return Cosine_Similarity


if __name__ == '__main__':
    # nltk.downloader.download_gui()
    nltk.data.path.append("nltk_data/")

    start = time.time()
    data_dic = {}
    i = 0

    f2 = open("../content/Stopword-List.txt", 'r')
    StopWords = f2.read()
    StopWords = StopWords.split("\n")
    f2.close()

    main_dir = os.listdir('../content/bbcsport')  # folders like: athletics,tennis
    for file in main_dir:
        sub_dir = os.listdir("../content/bbcsport/" + file)
        for filenum in sub_dir:
            data_dic[i] = {"id": filenum, "sport": file, "terms": tokenization(file, filenum, StopWords)
            # id:01.txt , sport: athletics, terms: [play, score, ......
            i = i + 1

    keys = list(data_dic.keys())  # List of keys to shuffle data
    random.shuffle(keys)

    #split data_dic in 30% and in 70%
    data_train, data_test = train_test_split(keys, test_size=0.3)

    BOW = BagOfWords(data_train, data_dic)

    training_tf = TermFrequency(data_train, data_dic, BOW)
    tf_idf = TF_IDF(data_train, training_tf, BOW)

    testing_tf = TermFrequency(data_test, data_dic, BOW)
    Query_tf_idf = TF_IDF(data_train, testing_tf, BOW)


    accuracy = 0
    outfile = open("../content/Classes.json", "w")

    for test_key in data_test: #testing set loop
        cosine_similarity = {}
        i=-1
        doc_id = data_dic[test_key]['sport'] + ": " + data_dic[test_key]['id']
        for train_key in data_train: #training set loop
            i = i + 1
            doc_id1 = data_dic[train_key]['sport'] + ": " + data_dic[train_key]['id']
            doc_id2 = str(i) + ":" +data_dic[test_key]['sport'] + "-" + data_dic[train_key]['sport']
            # number:test_file-training_file = 1:rugby-tennis

            cosine_similarity[doc_id2] = CosineSimilarity(training_tf[doc_id1], testing_tf[doc_id])

        cosine_similarity = dict(sorted(cosine_similarity.items(), key=operator.itemgetter(1), reverse=True))

        toplist = []
        for i, k in enumerate(cosine_similarity.keys()):
            if (i<3):
                k= k.split(":")
                toplist.append(k[1])

        try:
          class1 = mode(toplist)
        except StatisticsError:
          class1 = toplist[0]

        json_object = json.dumps(class1, indent=0)
        outfile.write(json_object + "\n")

        class1 = class1.split("-")
        if (class1[0] == class1[1]):
            accuracy +=1
    outfile.close()
    percentage = (accuracy/len(data_test))*100
    print(percentage)
    outfile.close()

