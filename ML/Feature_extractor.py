#!/usr/local/bin/python2.7
# encoding: utf-8
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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
        
    def get_contents(self):
        self.review_list = []
        with open('../Data/YelpZip/reviewContent','rb') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
         
            for row in tsvin:
                self.review_list.append(row[3])
                         
        print "Get-Contents done"

    def vectorize(self,ngram):
        vectorizer = CountVectorizer(min_df=0.01,ngram_range=(ngram,ngram),analyzer='word',stop_words='english',max_features=1000,token_pattern=r"(?u)\b[a-zA-Z]+'?[a-zA-Z]+\b",strip_accents='ascii')
        vectorizer.stop_words += ''
        
        self.X = vectorizer.fit_transform(self.review_list).toarray()
        self.names = vectorizer.get_feature_names()
        print len(self.names)
        
        print "Vectorize done"
        self.print_X()

    def transform(self):
        transformer = TfidfTransformer(smooth_idf=False)
        self.X = transformer.fit_transform(self.X).toarray()
        
        print "Get-Contents done"
        self.print_X()
        
    def save(self, fileName):
        with open('../Data/Extracted_features/'+fileName, 'wb') as r:
            r = csv.writer(r, delimiter='\t')
            
            r.writerow(self.names)
            for row in self.X:
                    r.writerow(row) 
        print "Saving done"     
    
    def print_X(self):
        print self.names
        print self.X
        