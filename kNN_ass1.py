'''
@author: Shichao Dong
'''
from collections import Counter
import math 
import numpy as np
from sklearn import cross_validation
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def sfold(lenth,n_fold):#my version of s-fold validation
    test =[]
    train =[]
    cvfold =[]
    count = int(math.ceil(float(lenth)/float(n_fold)))
    index = range(lenth)
    random.shuffle(index)
    stepindex = range(0,lenth,count)
    back_num = lenth - max(stepindex)
    for i in stepindex:
        if i != max(stepindex):
            temptrain = index[:i]
            temptrain += index[i+12:]
            temptest = index[i:i+12]
        else:
            temptrain = index[:i]
            temptest = index[i:]
            temptest = index[0:back_num]
        train.append(temptrain)
        test.append(temptest)
    cvfold.append(train)
    cvfold.append(test)
    return cvfold   

def euclidean(v1,v2): # The euclidean distance ,input are two vectors
    d=0.0
    for i in range(len(v1)): 
        d+=(float(v1[i])-float(v2[i]))**2 #the euclidean formula
    return math.sqrt(d)#return the distance from two vector

def KNN(k,train,test):# the main KNN function, the input are the number of k,the train set and the test set
    error = []#this is going to record the error state for each test,0=correct,1=error
    for t in test:#scan every element in the test data set
        tmap = {}#this is a dictionary structure to record the training vector's distance and its label{distance:class label}
        t = list(t)#transfer t to an list
        for tr in train:#scan every train vector
            tr = list(tr)
            dis = euclidean(t[:len(t)-1],tr[:len(tr)-1])#get the distance
            tmap[dis] = tr[-1]#training vector's distance and its label
        dict = sorted(tmap.iteritems(), key=lambda d:d[0])#sort the vector by distance
        temp =[]
        for i in range(k):#get the nearest k vector
            temp.append(dict[i][1])#get its class label
        label = Counter(temp).most_common(1)[0][0]# count the frequency of the group
        if label == t[-1]:# to see if the classification is correct
            error.append(0)#right
        else:
            error.append(1)#incorrect
    return error#return error list  

def processknn(k,S):
    url = 'bezdekIris.data'#the file's url
    data = np.genfromtxt(open(url,'r'), delimiter=',',dtype=None)#loading data
    cv = cross_validation.KFold(len(data), n_folds= S, indices=True)#the corss validation from library
    #cv = sfold(len(data), n_fold = S)# my s-fold validation
    results = []   
    for traincv, testcv in cv: #get the train and test data
        train =[]
        test = []
        for i in range(len(traincv)):
            train.append(data[traincv[i]])#train data
        for i in range(len(testcv)):
            test.append(data[testcv[i]])# test data
        error = KNN(k,train,test)#run the knn
        results.append(sum(error))#get the total error
    '''#my version of s-fold validation
    for t in range(S-1):
        train =[]
        test = []
        for i in range(len(cv[0][t])):
            train.append(data[cv[0][t][i]])
        for i in range(len(cv[1][t])):
            test.append(data[cv[1][t][i]])
        error = KNN(k,train,test)
        results.append(sum(error))       
    '''
    return min(results)

def drawplot(X,Y,Z):#draw the error curve
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(X,Y,Z, label='Minimal error curve')
    ax.set_xlabel("S from S-fold")
    ax.set_ylabel("k from k-NN")
    ax.set_zlabel("Minimal error")
    ax.legend()
    plt.show()

if __name__ == '__main__':
    K = range(1,40,2)#do the test from assignment
    S = [2,5,10]
    mini = []
    x = []#prepare to draw the plot
    y = []
    for s in S:
        for k in K:
            x.append(s)
            y.append(k)
            mini.append(int(processknn(k,s)))
    indexrange = range(0,61,20)#devide the error into 3 subset
    error = []
    minik = [] 
    for i in range(len(S)):#get the index of k with min error and different S
        error.append(mini[indexrange[i]:indexrange[i+1]])#devide the error into 3 subset
        error[i] = list(reversed(error[i]))#we want the largest k,so reverse the vector 
        minik.append(error[i].index(min(error[i])))#the first min error is the largest reversed index
        print min(error[i])
    for i in minik:
        print K[len(K)-i-1]#correct reversed index
    drawplot(x,y,mini)