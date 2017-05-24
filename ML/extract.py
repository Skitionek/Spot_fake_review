'''
Created on May 25, 2017

@author: s163601
'''
from  Feature_extractor import Feature_extractor

if __name__ == '__main__':
    pass

    ext = Feature_extractor(0)
    
    ext.get_contents()
    ext.vectorize(1)
    ext.transform()
    ext.save("Unigram")
    
    ext.vectorize(2)
    ext.transform()
    ext.save("Bi-gram")
    
    ext.vectorize(3)
    ext.transform()
    ext.save("Tri-gram")