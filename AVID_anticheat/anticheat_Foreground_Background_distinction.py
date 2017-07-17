# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA, FactorAnalysis,KernelPCA
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction import DictVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

from anticheat_common import k_means,k_means2,k_means_kernal_pca,noveltyDetection,dbscan,mahalanobisDistances
from anticheat_common import normalization


#加载数据
#样例数据（）1138f910700444eeb5cef2d4137cfe05,2017-01-18 09:39:57,600020,_CI,4
def loadDataSet(file):
    """
    保存成dic形式，dic中的元素为dic
    """
    dataDic = {}
    fr = open(file)
    for line in fr.readlines():
        dic = {}
        lineArr = line.strip().split(',')
        idKey = lineArr[0]
        typeKey = lineArr[3]
        
        if dataDic.has_key(idKey):
            dic = dataDic[idKey]
            if dic.has_key(typeKey):
                value = dic[typeKey]
                dic[typeKey] = value + 1
            else:
                dic[typeKey] = 1
        else:
            dic[typeKey] = 1
        
        dataDic[idKey]=dic
    fr.close()
    return dataDic

def loadUser(path):
    userDic = {}
    fr = open(path)
    for line in fr.readlines():
        userDic[line.strip().split(',')[1]]=line.strip().split(',')[0]
    return userDic

def preprocess(dic):
    """
    在用户的行为中增加前台行为和后台行为的统计。
    """
    data = {}
    for key in dic:
        value = dic[key]
        _All = 0
        All = 0
        for miniKey in value:
            if miniKey[0] == '_':
                _All = _All + value[miniKey]
            else:
                All = All + value[miniKey]
        value['_ALL'] = _All
        value['ALL'] = All
        value['rate'] = _All*1.0/(_All + All)
        data[key] = value
    return data

def preprocessV2(dic):
    """
    在用户的行为中增加前台行为和后台行为的统计。
    """
    data = {}
    for key in dic:
        value = dic[key]
        newvalue = {}
        _All = 0
        All = 0
        for miniKey in value:
            if miniKey[0] == '_':
                _All = _All + value[miniKey]
            else:
                All = All + value[miniKey]
        newvalue['_ALL'] = _All
        newvalue['ALL'] = All
        newvalue['rate'] = _All*1.0/(_All + All)
        data[key] = newvalue
    return data


def makeDataMat(dic):
    """
    把用户id和数据dic分开
    """
    idMat = []
    dataMat = []
    for key in dic:
        dataMat.append(dic[key])
        idMat.append(key)
    return idMat, dataMat

def makeVec(mat):
    """
    把用户行为字典变为向量
    """
    vec = DictVectorizer()
    dataVec = vec.fit_transform(mat).toarray()
    print dataVec.shape
    print dataVec[0]
    return dataVec

def makeTrain(dic, cheatUser, uncheatUser):
    label = []
    data = []
    for key in dic:
        if key in cheatUser:
            #print key
            dic[key][cheatUser[key]]=1
            label.append(0)
            data.append(dic[key])
        elif key in uncheatUser:
            #print key
            dic[key][uncheatUser[key]]=1
            label.append(1)
            data.append(dic[key])
    x = makeVec(data)
    return x, label

def LogisticRegression(data,label,pred_data,pred_last):
    '''
    效果不错，不过还是分类，需要正规化
    '''
    data=np.array(data)
    pred_data=np.array(pred_data)
    label=np.array(label)
    pred_last=np.array(pred_last)
    from sklearn.linear_model import LogisticRegression
    for i, C in enumerate(10. ** np.arange(1, 4)):
        # turn down tolerance for short training time
        clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
        clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)
        clf_l1_LR.fit(data,label)
        clf_l2_LR.fit(data,label)
        print clf_l1_LR.score(data,label)
        pred_result=clf_l1_LR.predict(pred_data)
        print("Number of mislabeled points out of a total %d points : %d" % (pred_data.shape[0],(pred_last != pred_result).sum()))
        print clf_l1_LR.score(pred_data,pred_last)

        print clf_l2_LR.score(data,label)
        pred_result=clf_l2_LR.predict(pred_data)
        print("Number of mislabeled points out of a total %d points : %d" % (pred_data.shape[0],(pred_last != pred_result).sum()))
        print clf_l2_LR.score(pred_data,pred_last)

    clf_l1_LR = LogisticRegression(C=1, penalty='l1', tol=0.01)
    clf_l1_LR.fit(data,label)
    print clf_l1_LR.score(data,label)
    pred_result=clf_l1_LR.predict(pred_data)
    print("Number of mislabeled points out of a total %d points : %d" % (pred_data.shape[0],(pred_last != pred_result).sum()))
    print clf_l1_LR.score(pred_data,pred_last)
    print pred_result
    return pred_result

def randomForest(data,label,pred_data,pred_last):
    '''不需要规范化
    '''
    data=np.array(data)
    pred_data=np.array(pred_data)
    label=np.array(label)
    pred_last=np.array(pred_last)
    clf=RandomForestClassifier(n_estimators=20, max_depth=None,min_samples_split=2)
    clf=clf.fit(data,label)
    print clf.score(data,label)
    pred_result=clf.predict(pred_data)
    print clf.score(pred_data,pred_last)
    print pred_result
    print pred_result.shape
    print clf.feature_importances_
    print pred_result
    print("Number of mislabeled points out of a total %d points : %d" % (pred_data.shape[0],(pred_last != pred_result).sum()))
    #np.savetxt("distinguish_output2.txt",pred_result,fmt="%f",delimiter="\t")

    scores = cross_val_score(clf, data, label,cv=5)
    #joblib.dump(clf,'zhaihuixin/zhai.pkl')
    return pred_result
    #print scores.mean()
    #print scores

def randomForestRegressor(data,label,pred_data,pred_last):
    '''不需要规范化
    '''
    data=np.array(data)
    pred_data=np.array(pred_data)
    label=np.array(label)
    pred_last=np.array(pred_last)
    clf=RandomForestRegressor(n_estimators=20)
    #print clf
    clf.fit(data,label)
    print clf.score(data,label)
    pred_result=clf.predict(pred_data)
    print clf.score(pred_data,pred_last)
    scores = cross_val_score(clf, data, label,cv=5)
    print np.rint(pred_result)
    print pred_last
    print clf.feature_importances_
    print pred_result
    #np.savetxt("outputrandomForestRegressor.txt",pred_result,fmt="%d",delimiter="\t")
    #return pred_result

def extraTreesClassifier(data,label):
    data=np.array(data)
    label=np.array(label)
    clf=ExtraTreesClassifier(n_estimators=120, max_depth=None,min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, data, label)
    clf=clf.fit(data, label)
    print scores.mean()

def GaussianNaiveBayes(data,label,pred_data,pred_last):
    '''not good
    '''
    data=np.array(data)
    pred_data=np.array(pred_data)
    label=np.array(label)
    pred_last=np.array(pred_last)
    from sklearn.naive_bayes import GaussianNB
    gnb=GaussianNB()
    clf=gnb.fit(data,label)

    print clf.score(data,label)
    pred_result=clf.predict(pred_data)
    print("Number of mislabeled points out of a total %d points : %d" % (pred_data.shape[0],(pred_last != pred_result).sum()))
    print clf.score(pred_data,pred_last)
    return pred_result

def MultinomialNaiveBayes(data,label,pred_data,pred_last):
    '''not good，不能规范化
    '''
    data=np.array(data)
    pred_data=np.array(pred_data)
    label=np.array(label)
    pred_last=np.array(pred_last)
    from sklearn.naive_bayes import MultinomialNB
    gnb=MultinomialNB()
    gnb.fit(data,label)

    print gnb.score(data,label)
    pred_result=gnb.predict(pred_data)
    print("Number of mislabeled points out of a total %d points : %d" % (pred_data.shape[0],(pred_last != pred_result).sum()))
    print gnb.score(pred_data,pred_last)
    print pred_result
    return pred_result

def GaussianProcesses(data,label,pred_data,pred_last):
    '''memory not enough
    '''
    data=np.array(data)
    pred_data=np.array(pred_data)
    label=np.array(label)
    pred_last=np.array(pred_last)
    from sklearn.gaussian_process import GaussianProcess
    gp=GaussianProcess(theta0=5e-1)
    print data
    print data.shape
    print label
    print label.shape
    gp.fit(data[:,8:14],label)

    return data

def LDA(data,label,pred_data,pred_last):
    '''not good，不需要规范化
    '''
    data=np.array(data)
    pred_data=np.array(pred_data)
    label=np.array(label)
    pred_last=np.array(pred_last)
    from sklearn.lda import LDA
    gnb=LDA()
    gnb.fit(data,label)

    print gnb.score(data,label)
    pred_result=gnb.predict(pred_data)
    print("Number of mislabeled points out of a total %d points : %d" % (pred_data.shape[0],(pred_last != pred_result).sum()))
    print gnb.score(pred_data,pred_last)
    return pred_result

def OutputCodeClassifier(data,label,pred_data,pred_last):
    '''
    0.76473194506
    Number of mislabeled points out of a total 841 points : 211
    0.749108204518
    需要规范化
    '''
    data=np.array(data)
    pred_data=np.array(pred_data)
    label=np.array(label)
    pred_last=np.array(pred_last)
    from sklearn.multiclass import OutputCodeClassifier
    from sklearn.svm import LinearSVC
    clf = OutputCodeClassifier(LinearSVC(random_state=0),code_size=2, random_state=0)
    clf.fit(data,label)

    print clf.score(data,label)
    pred_result=clf.predict(pred_data)
    print("Number of mislabeled points out of a total %d points : %d" % (pred_data.shape[0],(pred_last != pred_result).sum()))
    print clf.score(pred_data,pred_last)
    return pred_result

def BayesianRidgeRegression(data,label,pred_data,pred_last):
    '''
    效果很差
    '''
    data=np.array(data)
    pred_data=np.array(pred_data)
    label=np.array(label)
    pred_last=np.array(pred_last)
    from sklearn.linear_model import BayesianRidge, LinearRegression
    clf = BayesianRidge(compute_score=True)
    clf.fit(data,label)
    print clf.score(data,label)
    pred_result=clf.predict(pred_data)
    print("Number of mislabeled points out of a total %d points : %d" % (pred_data.shape[0],(pred_last != pred_result).sum()))
    print clf.score(pred_data,pred_last)


    ols = LinearRegression()
    ols.fit(data,label)
    print ols.score(data,label)
    pred_result=ols.predict(pred_data)
    print("Number of mislabeled points out of a total %d points : %d" % (pred_data.shape[0],(pred_last != pred_result).sum()))
    print ols.score(pred_data,pred_last)
    return pred_result




if __name__=="__main__":
    #加载数据文件
    filename = "/Users/holazhai/Documents/data/s3data/20170119_600025.log"
    dataDic = loadDataSet(filename)
    print len(dataDic)
    preData = preprocess(dataDic)
    idMat, dataMat = makeDataMat(preData)
    #print dataMat
    dataVec = makeVec(dataMat)
    data_ori = np.concatenate((dataVec[:3600,:],dataVec[3601:,:]))
    data = normalization(data_ori)

    #k_means2(data,1)
    #k_means(data,1)
    #k_means_kernal_pca(data,1)
    #dbscan(data)
    #noveltyDetection(data)
    #mahalanobisDistances(data)


    #加载作弊和非作弊用户列表
    cheatPath = "/Users/holazhai/Documents/IPython2/AVID_anticheat/cheatUser.csv"
    uncheatPath = "/Users/holazhai/Documents/IPython2/AVID_anticheat/uncheatUser.csv"
    
    cheatUser = loadUser(cheatPath)
    uncheatUser = loadUser(uncheatPath)

    x, y = makeTrain(preData, cheatUser, uncheatUser)
    x_nor = normalization(x)

    x_nor_train = x_nor[:1800]
    y_nor_train = y[:1800]
    x_nor_test = x_nor[1800:]
    y_nor_test = y[1800:]

    x_train = x[:1800]
    y_train = y[:1800]
    x_test = x[1800:]
    y_test = y[1800:]

    LogisticRegression(x_nor_train, y_nor_train, x_nor_test, y_nor_test)
    randomForest(x_train, y_train, x_test, y_test)
    randomForest(x_nor_train, y_nor_train, x_nor_test, y_nor_test)
    #randomForestRegressor(x_train, y_train, x_test, y_test)
    extraTreesClassifier(x_train, y_train)
    GaussianNaiveBayes(x_train, y_train, x_test, y_test)
    GaussianNaiveBayes(x_nor_train, y_nor_train, x_nor_test, y_nor_test)


    #k_means2(x_nor,0.02)
    #k_means(x_nor,0.02)
    #k_means_kernal_pca(x_nor,0.02)
    #noveltyDetection(x_nor)
    #mahalanobisDistances(x_nor)

    plotData(x_nor,y,0.02)

#new
filename = "/Users/holazhai/Documents/data/s3data/20170119_600025.log"
dataDic = loadDataSet(filename)
print len(dataDic)
preData = preprocessV2(dataDic)
idMat, dataMat = makeDataMat(preData)
#print dataMat
dataVec = makeVec(dataMat)
data_ori = np.concatenate((dataVec[:3600,:],dataVec[3601:,:]))
data = normalization(data_ori)

