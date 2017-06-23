#coding: utf-8
from word2vec import *
import sys
import os

if __name__ == '__main__':
    print 'loading model...'
    wv = WordVector()
    wv.loadWVModel('../model/sogou_6_2_500.model')
    print 'done'
    while True:
        query = raw_input("input >>> ")
        if query == 'exit' or query == 'quit':
            sys.exit(0)
        with open('household_query.txt', 'w') as ofs:
            ofs.write('0' + '\t' + query)
        generate_svm_samples('household_query.txt', '../data/household_keywords.txt', 'household_sample.txt', wv)
        os.system('svm-predict household_sample.txt ../model/household_training_data.model household_predict.txt')
        with open('household_predict.txt') as ifs:
            pred = int(ifs.read())
            if pred == 1:
                print 'answer >>> 要把空调温度调高么？'
            elif pred == 2:
                print 'answer >>> 要把空调温度调低么？'
            elif pred == 3:
                print 'answer >>> 要电视换台么？'
            else:
                print 'answer >>> 我不太明白你的意思'
        os.system('rm household_query.txt household_sample.txt household_predict.txt')

