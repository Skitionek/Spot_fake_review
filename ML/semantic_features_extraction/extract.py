import sys

'''
Created on May 25, 2017

@author: s163601
'''
from Feature_extractor import Feature_extractor

if __name__ == '__main__':
    pass
    
    try:
        feature_n = int(sys.argv[1].strip())
    except: 
        feature_n = 1000
        pass
    
    try:
        stemmer_idx = int(sys.argv[2].strip())
    except: 
        stemmer_idx = None
        pass
    
    
    ext = Feature_extractor(0)
    ext.get_contents(0)
        
    def analyze ():
        ext.vectorize(1,feature_n)
        ext.transform()
        #     ext.tfidtransform()
        ext.save("unigram")
        
        ext.vectorize(2,feature_n)  
        ext.transform()
        #     ext.tfidtransform()
        ext.save("bigram")  
            
    if (stemmer_idx == None):
        print("Number of features: ",feature_n," \nTIP: you can pass number of features and stemmer index by arguments.")
             
        analyze()
        
        print("Basic work done (can press ctrl-C) reaped analysis with pre-processing")
    
        for i in range(7):
            ext.setpreprocess(i)
            analyze()
    
    
    ext.setpreprocess(stemmer_idx)
    analyze()
    
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
