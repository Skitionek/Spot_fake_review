#!/usr/local/bin/python2.7
# encoding: utf-8
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import itertools
import numpy as np

'''
Created on May 25, 2017

@author: Dominik Maszczyk
'''

class Feature_extractor(object):
    '''
    classdocs
    '''

    def __init__(self, verbose):
        self.verbose = verbose
        '''
        Constructor
        '''
        
    def get_contents(self,fake):
        self.review_list = []

        with open('../Data/YelpZip/reviewContent','rb') as reviews, open('../Data/YelpZip/metadata','rb') as metadatas:
            reviews = csv.reader(reviews, delimiter='\t')
            metadatas = csv.reader(metadatas, delimiter='\t')
         
            for review,metadata in itertools.izip(reviews, metadatas):
                if (metadata[3] == str(-fake)):
                    self.review_list.append(review[3])
                         
        print "Get-Contents done"

    def vectorize(self,ngram):
        vectorizer = CountVectorizer(max_df=0.95,min_df=10,ngram_range=(ngram,ngram),analyzer='word',stop_words='english',max_features=100,token_pattern=r"(?u)\b[a-zA-Z]+'?[a-zA-Z]+\b",strip_accents='ascii')
        
        self.X = vectorizer.fit_transform(self.review_list).toarray()
        self.names = vectorizer.get_feature_names()
        print len(self.names)
        
        print "Vectorize done"
        self.print_X()

    def transform(self):
        transformer = TfidfTransformer(smooth_idf=False)
        self.X = transformer.fit_transform(self.X).toarray()
        
        print "Transform done"
        self.print_X()
        
    def save(self, fileName):
        with open('../Data/Extracted_features/'+fileName+'.npy', 'wb') as r:
           
            np.save(r, np.array(self.X))
            
        print "Saving done"     
    
    def print_X(self):
        print self.names
        print self.X
        