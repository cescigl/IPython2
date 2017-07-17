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

#加载数据
#样例数据（）1138f910700444eeb5cef2d4137cfe05,2017-01-18 09:39:57,600020,_CI,4
def loadDataSetList(file):
    """
    保存成dic形式，dic中的元素为list
    """
    dataDic = {}
    fr = open(file)
    for line in fr.readlines():
        l=[]
        lineArr = line.strip().split(',')
        temp = [lineArr[1], lineArr[3]]
        idKey = lineArr[0]
        if dataDic.has_key(idKey):
            l = dataDic[idKey]
        l.append(temp)
        dataDic[idKey]=l
    fr.close()
    return dataDic

def loadDataSetCnt(file):
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

def preprocess(dic, n):
    """
    根据每个用户的行为顺序，找出用户的连续n个行为key
    """
    result = {}
    for key in dic:
        l = dic[key]
        l.sort()
        
        temp = []
        for i in l:
            temp.append(i[1])
        t2={}
        for c in range(1, len(temp)):
            #print temp[c]
            tempKey = '#'.join(temp[c-1:c+n-1])
            if result.has_key(key):
                t2 = result[key]
                if t2.has_key(tempKey):
                    t2[tempKey] = t2[tempKey] + 1
                else:
                    t2[tempKey] = 1
            else:
                t2[tempKey] = 1
            result[key]=t2
    return result


def unionAllData2(dic1,dic2,dic3):
    """
    把用户的行为统计、2-gram，3-gram数据联合起来
    """
    dic = {}
    for key in dic1:
        result = {}
        temp = {}
        d1 = dic1[key]
        d2 = dic2[key] if dic2.has_key(key) else {}
        d3 = dic3[key] if dic3.has_key(key) else {}
        temp = dict(d1, **d2)
        result = dict(temp, **d3)
        dic[key] = result
        
    return dic

def unionAllData(dic1,dicList):
    """
    把用户的行为统计、2-gram，3-gram数据联合起来
    """
    dic = {}
    for key in dic1:
        result = dic1[key]
        for dl in dicList:
            dd = dl[key] if dl.has_key(key) else {}
            result = dict(result, **dd)
        dic[key] = result
        
    return dic

def makeDataMat(dic):
    """
    把用户字典中的用户id和特征分开
    """
    idMat = []
    dataMat = []
    for key in dic:
        dataMat.append(dic[key])
        idMat.append(key)
    return idMat, dataMat

def makeVec(mat):
    vec = DictVectorizer()
    dataVec = vec.fit_transform(mat).toarray()
    print dataVec.shape
    #print dataVec[0]
    return dataVec

def loadUser(path):
    """
    加载当日新增用户的渠道信息
    """
    userDic = {}
    fr = open(path)
    for line in fr.readlines():
        userDic[line.strip().split(',')[1]]=line.strip().split(',')[0]
    return userDic

def splitTrainTest(user, dataVec, ud):
    """
    把新增用户划分成训练集和测试集
    """
    train = []
    test = []
    for i in range(len(user)):
        if ud.has_key(user[i]):
            if ud[user[i]] == 'Facebook Ads' or ud[user[i]] == 'Organic' or ud[user[i]] == 'googleadwords_int':
                train.append(dataVec[i])
            else:
                test.append(dataVec[i])
    
    return train, test

#规范化数据
def scalerMaxMin(dm):
    '''将属性缩放到一个指定的最大和最小值（通常是1-0）之间'''
    X_train = np.array(dm)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    return X_train_minmax
    #将相同的缩放应用到测试集数据中
    #X_test = np.array([[ -3., -1.,  4.]])
    #X_test_minmax = min_max_scaler.transform(X_test)

def scale(dm):
    '''    公式为：(X-mean)/std  计算时对每个属性/每列分别进行。
    使用sklearn.preprocessing.StandardScaler类，使用该类的好处在于可以保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据。
    '''
    X_train = np.array(dm)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scale=scaler.transform(X_train)
    return X_train_scale
    #可以直接使用训练集对测试集数据进行转换
    #scaler.transform([[-1.,  1., 0.]])

def normalization(dm):
    '''正则化的过程是将每个样本缩放到单位范数（每个样本的范数为1）'''
    X_train = np.array(dm)
    normalizer = preprocessing.Normalizer().fit(X_train)  # fit does nothing
    X_train_norm=normalizer.transform(X_train)
    return X_train_norm, normalizer
    #可以直接使用训练集对测试集数据进行转换
    normalizer.transform([[-1.,  1., 0.]])

def noveltyDetection(train, test):
    train = np.array(train)
    test = np.array(test)
    pca = PCA(n_components=2)
    pca.fit(train)
    X_train = pca.transform(train)
    X_test = pca.transform(test)
    
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size

    h=0.2
    x_min, x_max = X_train[:, 0].min()-h, X_train[:, 0].max()+h
    y_min, y_max = X_train[:, 1].min()-h, X_train[:, 1].max()+h
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.title("Novelty Detection")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
    
    s = 20
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s)
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s)
    
    plt.axis('tight')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()


# In[2]:
filename = "/Users/holazhai/Documents/IPython2/AVID_anticheat/20170118-22_600025.log"
dataDicList = loadDataSetList(filename)

dataDicCnt = loadDataSetCnt(filename)


preDataDic3gram = preprocess(dataDicList, 3)
preDataDic2gram = preprocess(dataDicList, 2)

#unionDic = unionAllData(dataDicCnt, [preDataDic2gram, preDataDic3gram])
unionDic = unionAllData(preDataDic2gram, [])
idMat, dataMat = makeDataMat(unionDic)
#print dataMat
dataVec = makeVec(dataMat)
print dataVec[0]

path = '/Users/holazhai/Documents/IPython2/AVID_anticheat/20170118-22_600025_user_channel.txt'
userDic = loadUser(path)
train, test = splitTrainTest(idMat, dataVec, userDic)
print np.array(train).shape
print np.array(test).shape

X_train_norm, normalizer = normalization(train)
X_test_norm = normalizer.transform(np.array(test))

noveltyDetection(X_train_norm, X_test_norm)
#k_means(data)
#k_means_kernal_pca(data)
#dbscan(dataVec)
#noveltyDetection(data)
#mahalanobisDistances(data)

#对细分结果再聚类（正样本）
#dm=loadDataSetPositive('result2.log')
#data=scale(dm)
#k_means2(data,dm,dmid)
#dbscan(data)




