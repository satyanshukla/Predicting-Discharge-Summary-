import cPickle
import collections
import string
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

data = cPickle.load(open('../saved_files/48hr_keywords.p', 'rb')) 
print len(data)
p = string.punctuation + ' ' #to remove punctuations and space from end and start
unique_keywords = {}

for keywords in data:
    #print count
    tokens = keywords.split('\n')
    
    if len(keywords) > 0:
        for token in tokens:
            fragment = token.split('"')
            label = fragment[1].strip(p)
            if label not in unique_keywords:
                unique_keywords[label] = 1
            else:
                unique_keywords[label] += 1

od = collections.OrderedDict(sorted(unique_keywords.items()))
print(od)                
store = []
for key in unique_keywords:
    if unique_keywords[key] >=500:
        store.append(key)

"""
labels = []
for i in range(len(data)):
    keywords = data[i]
    tokens = keywords.split('\n')
    all_labels = []
    if len(keywords) > 0:
        for token in tokens:
            fragment = token.split('"')
            label = fragment[1].strip(p)
            if label in store:
                all_labels.append(label)
    #print len(all_labels)
    labels.append(all_labels)

mlb = MultiLabelBinarizer()
labels = np.array(labels)
labels = mlb.fit_transform(labels)
print labels.shape
print mlb.classes_
np.save('classes.npy', mlb.classes_)
#np.save('note_labels.npy', labels) 
values = np.sum(labels, axis=1)                                                                   
print np.histogram(values, bins=[0,10,20,30,40,50,60,70,80,90,100, 500, 1000])

"""


"""
labels = np.zeros((len(data), len(store)))
print labels.shape


for i in range(len(data)):
    print i
    
    for keyword in data[i]:
        if keyword in store:
            index = store.index(keyword)
            labels[i,index] = 1

for i in range(labels.shape[0]):
    print(np.where(labels[i] == 1)[0])
np.save('note_labels.npy', labels)
"""
"""

labels = np.load('note_labels.npy')
for i in range(labels.shape[0]):                                                                   
    print(np.where(labels[i] == 1)[0])
values = np.sum(labels, axis=1)
print values.shape
print np.histogram(values, bins=[0,1,2,3,4,5,6,7,8,9,100, 500, 1000])
  
        


"""
"""
values = []
print len(data)
print len(unique_keywords)
for key in unique_keywords:
    values.append(unique_keywords[key])
print np.histogram(values, bins =[1, 100, 500, 1000, 2000, 5000, 20000, 30000, 40000, 50000])
"""

"""
for key, value in sorted(unique_keywords.iteritems(), key=lambda (k,v): (v,k)):
    print "%s: %s" % (key, value)
"""


"""
data = cPickle.load(open('label.p', 'rb'))
data2 = cPickle.load(open('label2.p', 'rb'))
"""
"""
count = 0
for i in range(len(data)-1):
    if data[i+1][1] - data[i][1] == 1:
        count+=1

print count, len(data)

count =0
for i in range(len(data2)-1):
    if data2[i+1][1] - data2[i][1] == 1:
        count+=1

print count, len(data2)
"""
"""
keywords = []
for item in data:
    keywords.append(item[0])
print len(keywords)
for item in data2:
    keywords.append(item[0])
print len(keywords)

print keywords[0]
cPickle.dump( keywords, open( "keywords.p", "wb" ) )
"""
