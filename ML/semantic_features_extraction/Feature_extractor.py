from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

import nltk
from nltk import stem
from nltk.corpus import stopwords


import os

'''
Created on May 25, 2017

@author: Dominik Maszczyk
'''

class Feature_extractor(object):
    '''
    classdocs
    '''
    
    stemmers = ['porter','wordnet','lancaster','snowball','isri','rslp','regexp']
    stemmer = None
    stemmer_name = None

    def __init__(self, verbose):
        self.verbose = verbose
        '''
        Constructor
        '''
        
    def get_contents(self,fake):
        self.review_list = np.loadtxt('../../Data/YelpZip/reviewContent', usecols=[3], dtype='string', delimiter='\t')
                                             
        print("Get-Contents done ",len(self.review_list))

    def setpreprocess(self,stemmer_idx):
        self.stemmer_name = self.stemmers[stemmer_idx]
        if (self.stemmers[stemmer_idx] == 'porter'):
            self.stemmer = stem.PorterStemmer()
        if (self.stemmers[stemmer_idx] == 'wordnet'):
            self.stemmer = stem.WordNetLemmatizer()
        if (self.stemmers[stemmer_idx] == 'lancaster'):
            self.stemmer = stem.LancasterStemmer()
        if (self.stemmers[stemmer_idx] == 'snowball'):
            self.stemmer = stem.SnowballStemmer()
        if (self.stemmers[stemmer_idx] == 'isri'):
            self.stemmer = stem.ISRIStemmer()
        if (self.stemmers[stemmer_idx] == 'rslp'):
            self.stemmer = stem.RSLPStemmer()
        if (self.stemmers[stemmer_idx] == 'regexp'):
            self.stemmer = stem.RegexpStemmer()
            

    def vectorize(self,ngram,feature_n):
        if (self.stemmer == None):
            self.vectorizer = CountVectorizer(max_df=0.95,min_df=10,ngram_range=(ngram,ngram),analyzer='word',stop_words='english',max_features=feature_n,token_pattern=r"(?u)\b[a-zA-Z]+'?[a-zA-Z]+\b",strip_accents='ascii')
        else:
            print("Preprocessing by: ",self.stemmer_name)
#             analyzer = CountVectorizer().build_analyzer()
            
#             stop = set(stopwords.words('english'))
#         
#             if(self.stemmer_name == 'wordnet'):
#                 def stemmed_words(doc):
#                     return (self.stemmer.lemmatize(w) for w in analyzer(doc))
#             else: 
#                 def stemmed_words(doc):
#                     return (self.stemmer.stem(w) for w not in stop for w in analyzer(doc))
            
            if(self.stemmer_name == 'wordnet'):
                def stem_tokens(tokens, stemmer):
                    stemmed = []
                    for item in tokens:
                        stemmed.append(stemmer.stem(item))
                    return stemmed
            else: 
                def stem_tokens(tokens, stemmer):
                    stemmed = []
                    for item in tokens:
                        stemmed.append(stemmer.stem(item))
                    return stemmed
                
            def tokenize(text):
                tokens = nltk.word_tokenize(text)
                stems = stem_tokens(tokens, self.stemmer)
                return stems
            ######## 
            
            self.vectorizer = CountVectorizer(tokenizer=tokenize, max_df=0.95,min_df=10,ngram_range=(ngram,ngram),analyzer='word',stop_words='english',max_features=feature_n,token_pattern=r"(?u)\b[a-zA-Z]+'?[a-zA-Z]+\b",strip_accents='ascii')

        
        
        
        self.vectorizer.fit(self.review_list)
        
        print("Dictionary: ",self.vectorizer.get_feature_names())
        
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
        dir = '../../Data/Extracted_features/'
        if (self.stemmer_name == None):
            dir+="nopreprocessing/"
        else:
            dir+=self.stemmer_name+"/"
        
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(dir+fileName+'.npy', 'wb') as r:
           
            np.save(r, np.array(self.X))
            
        print("Saving done")   
    
    def print_X(self):
#         print(self.names)
        print(self.X)
        