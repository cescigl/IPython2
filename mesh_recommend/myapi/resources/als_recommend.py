# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from common.redshiftpool import RedShift
from common.redispool import RedisCache
import implicit
import copy
from scipy.sparse import coo_matrix
import logging
import os


class ALSRec(object):
    """docstring for ALSRec"""
    def __init__(self):
        super(ALSRec, self).__init__()
        self.redshift = RedShift()
        self.redis = RedisCache()
        self.userLeft_factor = "rec_userLeft_factor"
        self.userRight_factor = "rec_userRight_factor"
        self.log_filename="/Users/holazhai/Documents/IPython2/mesh_recommend/myapi/log/als_recommend.txt" 
        self.log_format=' [%(asctime)s]   %(message)s' 
        #将日志文件格式化  
        import logging
        logging.basicConfig(format=self.log_format,datafmt='%Y-%m-%d %H:%M:%S %p',level=logging.DEBUG,filename=self.log_filename,filemode='a') 


    def loadData(self):
        """
        从redshift加载数据
        """
        sql = "select fromuserid,touserid,case when duration>60000 then 60000 else duration end from meshmatch_event_prod where eventtype in ('LIVING_HANGUP','LIVING_USER_HANGUP');"
        sqlResult = self.redshift.getAll(sql)
        logging.debug('loadData')
        return sqlResult
        
    def loadFriendData(self):
        """
        从redshift加载好友信息
        """
        sql = "select userid,oppositeuserid,70000 as cnt from meshmatch_friend_prod;"
        sqlResult = self.redshift.getAll(sql)
        logging.debug('loadFriendData')
        return sqlResult

    def getPercentile(self, arr, l):
        """
        计算分位数
        """
        return [np.percentile(arr, i) for i in l]

    def preProcessData(self):
        """
        预处理数据
        """
        friend = self.loadFriendData() #用户好友信息
        print len(friend)
        living = self.loadData() #用户通话详情，
        print len(living)

        raw = friend + living 

        rawArray = np.array(raw, dtype=int)
        temp = []
        for l in rawArray:
            if l[0] < l[1]:
                temp.append([l[0], l[1], l[2]])
            else:
                temp.append([l[1], l[0], l[2]])

        # 根据前两列分组，取最大值
        df = pd.DataFrame(temp, columns=['id1', 'id2', 'cnt'])
        dfGroupby = df.iloc[df.groupby(['id1', 'id2']).apply(lambda x: x['cnt'].idxmax())]
        scoreLeft = np.array(dfGroupby)
        #两列用户id顺序颠倒
        scoreRight = copy.deepcopy(scoreLeft)
        scoreRight[:, [0, 1]] = scoreRight[:, [1, 0]]
        score = np.concatenate((scoreLeft,scoreRight))
        logging.debug('preProcessData')
        return score

    def matrixData(self, score):
        data = pd.DataFrame(score, columns=['id1', 'id2', 'cnt'])
        data['id1'] = data['id1'].astype("category")
        data['id2'] = data['id2'].astype("category")

        living = coo_matrix((data['cnt'].astype(float),
                       (data['id1'].cat.codes.copy(),
                        data['id2'].cat.codes.copy())))
        return data, living

    def bm25_weight(self, X, K1=100, B=0.8):
        """ Weighs each row of the sparse matrix of the data by BM25 weighting """
        # calculate idf per term (user)
        X = coo_matrix(X)
        N = X.shape[0]
        #print (N)
        idf = np.log(float(N) / (1 + np.bincount(X.col)))
        #print (idf)
        # calculate length_norm per document (artist)
        row_sums = np.ravel(X.sum(axis=1))
        average_length = row_sums.mean()
        length_norm = (1.0 - B) + B * row_sums / average_length
    	
        # weight matrix rows by bm25
        X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
        return X

    def saveToRedis(self, data, user1_factors, user2_factors):
        """
        把用户向量保存在redis中
        """
        l = list(set(np.array(data)[:,0]))
        l.sort()
        for i in range(len(l)):
            print i
            self.redis.hset_hset(self.userLeft_factor, l[i], user1_factors[i].tostring())
            self.redis.hset_hset(self.userRight_factor, l[i], user2_factors[i].tostring())
        logging.debug('saveToRedis')

    def alsRec(self):
        score = self.preProcessData()
        data ,living = self.matrixData(score)
        weighted = self.bm25_weight(living)
        print weighted.shape
        user1_factors, user2_factors = implicit.alternating_least_squares(weighted, factors=5) 
        print "save to redis"
        self.saveToRedis(data, user1_factors, user2_factors)

def checkFun(pNum=1):
    """
    #执行检查当前进程数目
    """
    #cmd = "ps aux | grep 'python main_zhx.py' | grep -v 'grep' | wc -l"
    cmd = "ps aux | grep 'als_recommend.py' | grep -v 'grep' | wc -l"
    num = int(os.popen(cmd).readlines()[0])
    if (num > pNum):
        sys.exit('Process has running!')

if __name__ == '__main__':
    checkFun()
    alsrec = ALSRec()
    alsrec.alsRec()

