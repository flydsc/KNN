'''

@author: Local-Admin
'''
import math
import random 

def kfold(lenth,n_fold):
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

    

if __name__ == '__main__':
    cv = kfold(150,13)
    print len(cv[0][0])
    print len(cv[1][0])