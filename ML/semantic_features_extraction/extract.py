'''
Created on May 25, 2017

@author: s163601
'''
from Feature_extractor import Feature_extractor

if __name__ == '__main__':
    pass

    ext = Feature_extractor(0)
    
    ext.get_contents(0)
    ext.vectorize(1)
    
    ext.get_contents(-1)
    ext.transform()
    ext.tfidtransform()
    ext.save("Fake/unigram")
    ext.get_contents(1)
    ext.transform()
    ext.tfidtransform()
    ext.save("notFake/unigram")
    
    ext.get_contents(0)
    ext.vectorize(2)  
      
    ext.get_contents(-1)
    ext.transform()
    ext.tfidtransform()
    ext.save("Fake/bigram")
    ext.get_contents(1)
    ext.transform()
    ext.tfidtransform()
    ext.save("notFake/bigram")
    
    
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