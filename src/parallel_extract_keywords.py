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

for i in range(len(data)):
    data[i] = [data[i], i]

data = data[len(data)/2:]
print len(data)    
def my_function(item):
    return [predict(item[0], 'models/silver.crf', 'i2b2', use_lstm=True), item[1]]

"""
from multiprocessing.dummy import Pool as ThreadPool 
pool = ThreadPool(40) 
results = pool.map(my_function, data)
"""

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
print num_cores
results = Parallel(n_jobs=num_cores)(delayed(my_function)(item) for item in data)
cPickle.dump( results, open( "label2.p", "wb" ) )
