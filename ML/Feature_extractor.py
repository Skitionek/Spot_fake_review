#!/usr/local/bin/python2.7
# encoding: utf-8

import csv
from sklearn.feature_extraction.text import CountVectorizer

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

review_list = []
with open('../Data/tmp/Contents', 'wb') as tsvout:
    with open('../Data/YelpZip/reviewContent','rb') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
     
        for row in tsvin:
            tsvout.write(row[3])
            review_list.append(row[3])
                 
print("Contents done")

vectorizer = CountVectorizer(min_df=0.1,ngram_range=(1,2),analyzer='word',stop_words='english',max_features=1000,token_pattern=r"(?u)\b[a-zA-Z]+'?[a-zA-Z]+\b",strip_accents='ascii')
vectorizer.stop_words += ''

X = vectorizer.fit_transform(review_list).toarray()
print X
names = vectorizer.get_feature_names()
print names
print len(names)

print("Vectorizer done")

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
result = transformer.fit_transform(X).toarray()

print "Results what you would work on:"
print result

with open('../Data/Extracted_features', 'wb') as r:
    r = csv.writer(r, delimiter='\t')
    
    r.writerow(names)
    for row in result:
            r.writerow(row)      