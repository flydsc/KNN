'''

@author: Local-Admin
'''
from collections import Counter
import math 
import numpy as np
from sklearn import cross_validation
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def euclidean(v1,v2): 
    d=0.0
    for i in range(len(v1)): 
        d+=(float(v1[i])-float(v2[i]))**2 
    return math.sqrt(d)
 
def readdata(url):
    inputfile = open(url,'r')
    data = []
    for i in inputfile:
        temp = i.split(',')
        data += [temp]
    return data

def maketrain(data,lenth):
    traindata = []
    for i in data:
        count = len(i) / 2
        for c in range(count):
            traindata.append(i[c])
    return traindata

def maketest(data,lenth):
    testdata = []
    for i in data:
        count = len(i) / 2
        for c in range(count,len(i)):
            testdata.append(i[c])
    return testdata

def proccessdata(url):
    data = np.genfromtxt(open(url,'r'), delimiter=',')[:]
    train = data[:int(0.9*len(data))]
    test = data[int(0.9*len(data)):]
    process = []
    '''

    classlabel = []
    for i in data:
        classlabel += [i[-1]]
    label = set(classlabel)
    label = list(label)
    classdata = []
    for l in label:
        temp = []   
        for d in data:
            if l == d[-1]:
                temp.append(d)
        classdata.append(temp)
    classlenth = []
    for i in classdata:
        classlenth.append(len(i))
    train  = maketrain(classdata,classlenth)
    test =  maketest(classdata,classlenth)
    '''
    process.append(train)
    process.append(test)
    return process

def sortdict(adict):
    dict= sorted(adict.iteritems(), key=lambda d:d[0])
    return dict 

def KNN(k,train,test):
    error = []#0=correct,1=error
    for t in test:
        tmap = {}
        t = list(t)
        for tr in train:
            tr = list(tr)
            dis = euclidean(t[:len(t)-1],tr[:len(tr)-1])
            tmap[dis] = tr[-1]
        dict = sorted(tmap.iteritems(), key=lambda d:d[0])
        temp =[]
        for i in range(k):
            temp.append(dict[i][1])
        label = Counter(temp).most_common(1)[0][0]
        if label == t[-1]:
            error.append(0)
        else:
            error.append(1)
    return error  

def processknn(k,S):
    url = 'bezdekIris.data'
    data = np.genfromtxt(open(url,'r'), delimiter=',',dtype=None)
    cv = cross_validation.KFold(len(data), n_folds= S, indices=True)
    results = []
    for traincv, testcv in cv:
        train =[]
        test = []
        for i in range(len(traincv)):
            train.append(data[traincv[i]])
        for i in range(len(testcv)):
            test.append(data[testcv[i]])
        error = KNN(k,train,test)
        results.append(sum(error))
    '''
    print "The S is " + str(S) + "\tThe k is " + str(k) +":"
    print results
    print "The averrage is "+ str(sum(results)/len(results))
    
    '''
    return min(results)

def drawplot(X,Y,Z):
    print X
    print Y
    print Z
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(X,Y,Z, label='Minimal error curve')
    ax.set_xlabel("S from S-fold")
    ax.set_ylabel("k from k-NN")
    ax.set_zlabel("Minimal error")
    ax.legend()
    ax.legend()
    plt.show()

if __name__ == '__main__':
    K = range(1,40,2)
    S = [2,5,10]
    mini = []
    x = []
    y = []
    for s in S:
        for k in K:
            x.append(s)
            y.append(k)
            mini.append(int(processknn(k,s)))
    #
    indexrange = range(0,61,20)
    error = []
    minik = [] 
    for i in range(len(S)):
        error.append(mini[indexrange[i]:indexrange[i+1]])
        error[i] = list(reversed(error[i]))
        minik.append(error[i].index(min(error[i])))
        print min(error[i])
    for i in minik:
        print K[len(K)-i-1]
    drawplot(x,y,mini)

        
    