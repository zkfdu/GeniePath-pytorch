import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pandas as pd
import xlrd
from sklearn.model_selection import train_test_split
import pickle
import shap

fraud = pd.read_csv('fraud5.1.csv',nrows=5000)

b={k: list(d.index) for k, d in fraud.groupby('opposing_id') if len(d) > 0 }

c=b.values()
d=list(c)
print(d)
dict={}
for i in d:
    for l in i:
        i.remove(l)
        dict[l]=i
print(dict)

fraud=pd.get_dummies(fraud)


fraudfeature=fraud.drop(['dt','flag_case','amt'], axis=1)
#print(fraudfeature)
#print(fraudfeature.dtypes)
fraudlable=fraud.flag_case
#print(fraudlable)

#print(y)
#print(X)

X_train, X_test, Y_train, Y_test = train_test_split(fraudfeature,fraudlable, test_size=0.2, random_state=0,shuffle=False)

index_x=Y_test.index
list1=list(index_x)
print(list1)

X_train=X_train.astype('float64')
X_train=X_train.values
X_test=X_test.values
Y_train=Y_train.values
Y_train=Y_train.reshape(Y_train.shape[0],1)
Y_test =Y_test.values
ty=Y_test.reshape(Y_test.shape[0],1)
allx= sp.csr_matrix(X_train)
tx= sp.csr_matrix(X_test)
ally= Y_train


#print(allx.shape)
#print(ally.shape)
#print(tx)

allx_train, allx_test, ally_train, ally_test = train_test_split(X_train,Y_train , test_size=0.5, random_state=0,shuffle=False)
x_train=sp.csr_matrix(allx_train)
y_train=ally_train


fx= open("ind.fraud.x", "wb")
pickle.dump(x_train, fx)
fy= open("ind.fraud.y", "wb")
pickle.dump(y_train, fy)
fallx=open("ind.fraud.allx", "wb")
pickle.dump(allx, fallx)
fally=open("ind.fraud.ally", "wb")
pickle.dump(ally, fally)
ftx=open("ind.fraud.tx", "wb")
pickle.dump(tx, ftx)
fty=open("ind.fraud.ty", "wb")
pickle.dump(ty, fty)
x_index=open("ind.fraud.test.index", "wb")
pickle.dump(list1, x_index)
x_index.close()
graph=open("ind.fraud.graph", "wb")
pickle.dump(dict, graph)

