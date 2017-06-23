#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import gensim
import os
import sys
import math
import operator
from functools import partial
import jieba.analyse
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
import multiprocessing
import itertools
import re

class SentenceIter(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for filename in os.listdir(self.dirname):
            with open(os.path.join(self.dirname, filename)) as f:
                for line in f:
                    yield line.split()

class WordVector(object):

    def loadWVModel(self, modelDir):
        self.model = gensim.models.Word2Vec.load(modelDir)


    def dumpModel(self):
        print str(self.model)


    def train(self, corpusDir, modelDir):
        cpu_count = multiprocessing.cpu_count()
        sentIter = SentenceIter(corpusDir)
        self.model = gensim.models.Word2Vec(sentIter, sg=1, window=6, min_count=2, size=500, workers=cpu_count)
        self.model.save(modelDir)

    def most_similar(self, word):
        print ">>> %s" % (word)
#        word = word.decode('utf-8')
        try:
            for w,s in self.model.most_similar(word):
                print "%.6f %s" % (s, w)
        except:
            print "[WARN] low-frequency word"

    def similarity(self, word1, word2):
        return self.model.similarity(word1, word2)

    def findSimiliar(self, postiveList):
        semanticVector =  self.model.most_similar(postiveList)
        return semanticVector

    def levenshteinSimilarity(self, str1, str2):
        return 1.0 - normalized_damerau_levenshtein_distance(str1, str2)

    def semanticSimilarityJacard(self, sv1, sv2, withValue=True):
        if len(sv1) == 0 or len(sv2) == 0:
            return 0.0
        if not withValue:
            set1 = set(sv1)
            set2 = set(sv2)
            len1 = len(set1 & set2) * 1.0
            len2 = len(set1 | set2)
            return len1 / len2

        set1 = set(map(lambda x: x[0], sv1))
        set2 = set(map(lambda x: x[0], sv2))
        len1 = len(set1 & set2) * 1.0
        len2 = len(set1 | set2)
        return len1 / len2

    def semanticSimilarityCosine(self, sv1, sv2):
        dict1 = dict(sv1)
        dict2 = dict(sv2)
        partialPow = lambda x: pow(x, 2)
        norm1 = math.sqrt(reduce(operator.add, map(partialPow, dict1.values())))
        norm2 = math.sqrt(reduce(operator.add, map(partialPow, dict2.values())))
        innerProduct = 0.0
        for item in dict1.keys():
            innerProduct += dict1[item] * dict2.setdefault(item, 0.0)
        return innerProduct / (norm1 * norm2)


    def getKeyWords(self, str, topN=65535):
        candidates = jieba.analyse.extract_tags(str, topN)
        result = []
        for c in candidates:
            try:
                posList = []
                posList.append(c.encode('utf-8'))
                sv = self.findSimiliar(posList)
                result.append(c.encode('utf-8'))
            except Exception, e:
                print e
#                print c.encode('utf-8'), 'not found in Word2Vec'
        return result

    def doc_similarity_mean(self, str1, str2, topN=65535):
        """get normalized similarity between two sentences(words)"""
        keywords1 = jieba.analyse.extract_tags(str1, topN)
        keywords2 = jieba.analyse.extract_tags(str2, topN)
        sim_sum = 0.0
        for (k1, k2) in itertools.product(keywords1, keywords2):
            try:
                sim = max(0.0, self.similarity(k1.encode('utf-8'), k2.encode('utf-8')))
            except :
                pass
    def doc_similarity_max(self, str1, str2, topN=65535):
        """get max similarity between two sentences(words)"""
        keywords1 = isinstance(str1, str) and jieba.analyse.extract_tags(str1, topN) or str1
        keywords2 = isinstance(str2, str) and jieba.analyse.extract_tags(str2, topN) or str2
        return self.tag_list_similarity_max(keywords1, keywords2)

#==============================================================================
#        #DEBUG
#        print "keywords1:"
#        for word in keywords1:
#            print word,
#        print ''
#
#        print 'keywords2:'
#        for word in keywords2:
#            print word,
#        print ''
#==============================================================================
    def tag_list_similarity_max(self, tag_list1, tag_list2):
        sim = 0.0
        max_sim = 0.0
        for (k1, k2) in itertools.product(tag_list1, tag_list2):
            try:
                sim = max(0.0, self.similarity(k1.encode('utf-8'), k2.encode('utf-8')))
            except:
                sim = 0.0
            #DEBUG
#            print k1, k2, sim
            if sim > max_sim:
                max_sim = sim
#        inputs = raw_input('please press any key to continue')
        assert max_sim > -1.0, 'max_sim error'
        return max_sim


def load_features(fn):
    """load features"""
    feature_words = set()
    with open(fn) as ifs:
        for line in ifs:
            line = line.strip('\n')
            if not line:
                continue
            words = line.split(',')
            feature_words.update(words)
    return feature_words

def generate_svm_samples(corpus_fn, feature_fn, training_fn, word2vec_wrapper):
    """generate libsvm format training data by word2vec"""
    feature_words = load_features(feature_fn)

    with open(corpus_fn, 'r') as ifs:
        with open(training_fn, 'w') as ofs:
            for line_id, line in enumerate(ifs):
                line = line.strip('\n')
                label = line[0]
                content = line[2:]
                feature_vector = []

                for i, feature in enumerate(feature_words):
                    sim = word2vec_wrapper.doc_similarity_max(line, feature)
                    feature_vector.append(str(i+1) + ':' + str(sim))

                ofs.write(label + '\t' + ' '.join(feature_vector))
                ofs.write('\n')
