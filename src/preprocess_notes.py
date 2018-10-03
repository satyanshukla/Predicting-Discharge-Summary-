from __future__ import print_function
import numpy as np
import cPickle
from copy import deepcopy
timestamp = 200
num_features = 12

x = np.load('/home/snshukla/mimiciii/input_48hr_data.npy')
T = cPickle.load(open('/home/snshukla/mimiciii/timestamps_48hr_data.p', 'rb'))
output = cPickle.load(open('/home/snshukla/mimiciii/48hr_data_output.p', 'rb'))
notes = np.load('note_labels.npy')
values = np.sum(notes, axis=1)
#trim input data to remove subjects with timestamps higher than 200                                                                                               
remove = []
for i in range(len(T)):
        if len(T[i]) > timestamp or values[i]==0:
                remove.append(i)

x = np.delete(x, remove, axis = 0)
notes = np.delete(notes, remove, axis = 0)
x = x[:,:,:timestamp]
M = np.zeros_like(x)
delta = np.zeros_like(x)

for index in sorted(remove, reverse=True):
        del T[index]
        del output[index]
        

print(x.shape, len(T), len(output), notes.shape)

for t in T:
        for i in range(1,len(t)):
                t[i] = (t[i] - t[0]).total_seconds()/3600.0
        if len(t) != 0:
                t[0] = 0


avg = []
#### Calculating average for GRU mean                                                                                                                             
for i in range(num_features):
        sum = 0
        count = 0
        for j in range(x.shape[0]):
                for k in range(len(T[j])):
                        if x[j,i,k] == -1:
                                x[j,i,k] = 2
                        if x[j,i,k] <  0:
                                x[j,i,k] = -100 # count as missing data                                                                                           

                        if x[j,i,k] != -100:
                                sum = sum + x[j,i,k]
                                count += 1
                                M[j,i,k] = 1
                        if x[j,i,k] > 505:  #removing outliers
                                M[j,i,k] = 0

                        if k == 0:
                                delta[j,i,k] = 0
                        elif M[j,i,k-1] == 1:
                                delta[j,i,k] = T[j][k]
                        elif M[j,i,k-1] == 0:
                                delta[j,i,k] = T[j][k]
        avg.append(float(sum)/count)
print(avg)
for i in range(x.shape[0]):
        for j in range(num_features):
                for k in range(timestamp):
                        if x[i,j,k] == -100:
                                x[i,j,k] = avg[j]
                        if x[i,j,k] == 0: #logical thing to do unless 0 is a possible value which is not the case
                                M[i,j,k] = 0


#if a features is not observed at all, put a '1' in that row to help solve the problem o\f division by zero while interpolation, now the value for that row will fall to avg 

m1 = np.sum(M, axis = 2, dtype=np.int32)
for i in range(M.shape[0]):
        for j in range(M.shape[1]):
                if m1[i,j] == 0:
                        M[i,j,0] = 1
                        x[i,j,0] = avg[j]

#randomly dropping 15% of the observed values
rM = np.copy(M)
for i in range(M.shape[0]):
        for j in range(M.shape[1]):
                count = int(0.2*m1[i,j])
                if count > 1:
                        index = 0
                        r = np.ones((m1[i,j],1))
                        b = np.random.choice(m1[i,j], count, replace=False)
                        r[b] = 0
                        #r = np.random.choice(2, m1[i,j], p=[0.30,0.70])
                        for k in range(M.shape[2]):
                                if M[i,j,k] > 0:
                                        rM[i,j,k] = M[i,j,k]*r[index]
                                        index += 1

print(m1[0])
print(np.sum(M[0], axis = 1))
print(np.sum(rM[0], axis = 1))
z = np.zeros((x.shape[0], 4*num_features, timestamp))
for i in range(x.shape[0]):
        z[i] = np.vstack((x[i],M[i],delta[i],rM[i]))

x = []
M = []
delta = []
rM = []

x = z

count1 = 0
count2 = 0
for i in range(x.shape[0]):
        for j in range(x.shape[1]/3):
                for k in range(x.shape[2]):
                        if x[i,j,k] == 0:
                                count1 +=1
                        if x[i,j,k] < 0:
                                count2 +=1

z = []
#for i in range(x.shape[0]):                                                                                                                                      
#       x[i] = preprocessing.scale(x[i])                                                                                                                          


y = np.zeros((len(output),1), dtype = int)
for i in range(len(output)):
        y[i] = output[i]


inds = np.arange(x.shape[0])
np.random.seed(seed=0)
np.random.shuffle(inds)
x = x[inds]
y = y[inds]
notes = notes[inds]

np.save('input_for_notes.npy', x)
np.save('label_notes.npy', notes)
np.save('mortality_labels.npy', y)
print('Saving Done!')
print(x.shape, y.shape, notes.shape)
