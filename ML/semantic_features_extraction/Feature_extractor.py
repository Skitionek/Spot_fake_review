import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip


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
        self.review_list = np.loadtxt('../../Data/YelpZip/reviewContent',usecols=3, dtype='string', delimiter='\t')
                                             
        print("Get-Contents done ",len(self.review_list))

    def vectorize(self,ngram,feature_n):
        self.vectorizer = CountVectorizer(max_df=0.95,min_df=10,ngram_range=(ngram,ngram),analyzer='word',stop_words='english',max_features=feature_n,token_pattern=r"(?u)\b[a-zA-Z]+'?[a-zA-Z]+\b",strip_accents='ascii')
        
        self.vectorizer.fit(self.review_list)
        
        print("Vectorize done")

    def transform(self):
        
        self.X = self.vectorizer.transform(self.review_list).toarray()
        print("Transform done")
        print(self.X)

#     def tfidtransform(self):
#         transformer = TfidfTransformer(smooth_idf=False)
#         self.X = transformer.fit_transform(self.X).toarray()
#         
#         print("Tfidtransform done")
#         self.print_X()
        
    def save(self, fileName):
        with open('../../Data/Extracted_features/'+fileName+'.npy', 'wb') as r:
           
            np.save(r, np.array(self.X))
            
        print("Saving done")   
    
    def print_X(self):
#         print(self.names)
        print(self.X)
        