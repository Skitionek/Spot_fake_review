import sys
import os

'''
Created on May 25, 2017

@author: s163601
'''
from Feature_extractor import Feature_extractor

if __name__ == '__main__':
    pass
    
    try:
        feature_n = int(sys.argv[1].strip())
        stemmer_idx = int(sys.argv[2].strip())
    except: 
        feature_n = 1000
        stemmer_idx = None
        pass
    
        
    print("Number of features: ",feature_n," \nTIP: you can pass number of features and stemmer index by arguments.")
        
    ext = Feature_extractor(0)
    ext.get_contents(0)
    
    if (stemmer_idx!=None):
        ext.setpreprocess(0)
        ext.vectorize(1,100)
        ext.setpreprocess(1)
        ext.vectorize(1,100)
        ext.setpreprocess(2)
        ext.vectorize(1,100)
        ext.setpreprocess(3)
        ext.vectorize(1,100)
        ext.setpreprocess(4)
        ext.vectorize(1,100)
        ext.setpreprocess(5)
        ext.vectorize(1,100)
        ext.setpreprocess(6)
        ext.vectorize(1,100)
    
    def analyze (fol):
        ext.vectorize(1,feature_n)
        ext.transform()
    #     ext.tfidtransform()
        ext.save(fol+"unigram")
    
        ext.vectorize(2,feature_n)  
        ext.transform()
    #     ext.tfidtransform()
        ext.save(fol+"bigram")
        
    analyze()
    
    print("Basic work done (can press ctrl-C) reaped analysis with pre-processing")
    
    for i in range(7):
        ext.setpreprocess(i)
        path = str(i)+"/"
        if not os.path.exists(path):
            os.makedirs(path)
        analyze(path)
    
#     ext.get_contents(0)
#     ext.vectorize(3)
#     
#     ext.get_contents(-1)
#     ext.transform()
#     ext.tfidtransform()
#     ext.save("Fake/trigram")
#     ext.get_contents(1)   
#     ext.transform()
#     ext.tfidtransform()
#     ext.save("notFake/trigram")