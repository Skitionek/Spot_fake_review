'''
Created on May 25, 2017

@author: s163601
'''
from Feature_extractor import Feature_extractor

if __name__ == '__main__':
    pass

    ext = Feature_extractor(0)
    
    ext.get_contents(-1)
    ext.vectorize(1)
    ext.transform()
    ext.save("Fake/unigram")
    
    ext.vectorize(2)
    ext.transform()
    ext.save("Fake/bigram")
    
    ext.vectorize(3)
    ext.transform()
    ext.save("Fake/trigram")
    
    ext.get_contents(1)
    ext.vectorize(1)
    ext.transform()
    ext.save("notFake/unigram")
    
    ext.vectorize(2)
    ext.transform()
    ext.save("notFake/bigram")
    
    ext.vectorize(3)
    ext.transform()
    ext.save("notFake/trigram")