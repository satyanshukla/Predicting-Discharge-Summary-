import os
import sys
import cPickle

# Where to import code from                                                                      
homedir = os.path.dirname(os.path.abspath(__file__))
codedir = os.path.join(homedir, 'code')
if codedir not in sys.path:
    sys.path.append(codedir)
from predict2 import predict

data = cPickle.load(open('clinical_notes.p', 'rb'))
print len(data)
labels = []
count = 0
for item in data:
    count += 1
    print count
    label = predict(item, 'models/silver.crf', 'i2b2', use_lstm=True)
    labels.append(label)
    

print len(labels)
