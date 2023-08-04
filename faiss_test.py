import argparse
import json
import os
import shutil
from typing import List, Optional
import urllib

import numpy as np




d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

'''
print(xq)
print('*' * 100)
print(xb)
'''

a  = np.random.random((4,2)).astype('float32')
print(a)
print('-------------------------------------------')
a[:, 0] += np.arange(4)
print(a)
print('-------------------------------------------')

import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)


print(xb[:5])
print('-'*100)

k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
print('-' *100)
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(D[:5])   
#print(I[-5:])                  # neighbors of the 5 last queries
  
for j, i in enumerate(I[0]):
    print(f'j={j}, i= {i}')


def seperate_list(ls: List[int]) -> List[List[int]]:
    # TODO: 增加是否属于同一文档的判断
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


if __name__ == "__main__":
    print('-------------------------------------------')

    i = 3
    N = 10

    for k in range(1, max(i, N-i)):
        #print(k)
        #print(ia -k, ia+k)
        rg = [i+k]
        for j in rg:
            print(j)
            #print('-'*10)
    print('---'*10)
    l = [1,2,5,6,8,12,13]
    ret = seperate_list(l)
    print(ret)