# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import sys
sys.path.append("/data/mesh_push_service/meshapi/")
from sklearn import preprocessing
from common.redshiftpool import RedShift
from common.config import RedisDBConfig
import redis
import logging
import os


class Popular(object):
    """docstring for Popular"""
    def __init__(self):
        super(Popular, self).__init__()
        self.redshift = RedShift()
        self.userPopular = 'rec_popular' # 记录用户的流行度
        self.log_filename="/data/mesh_push_service/meshapi/log/popular_recommend.txt" 
        self.log_format=' [%(asctime)s]   %(message)s' 
        #将日志文件格式化  
        import logging
        logging.basicConfig(format=self.log_format,datafmt='%Y-%m-%d %H:%M:%S %p',level=logging.DEBUG,filename=self.log_filename,filemode='a') 

    def loadData(self):
        """
        从redshift加载数据
        """
        sql = "select userid,case when like_count>0 then like_count else 1 end,case when liked_count>0 then liked_count else 0 end,case when conn_count>0 then conn_count else 0 end,case when conn_avg>0 then conn_avg else 20000 end from meshmatch_user_ext;"
        sqlResult = self.redshift.getAll(sql)
        logging.debug('loadData')
        return sqlResult

    def saveToRedis(self, userid, data):
        r = redis.Redis(host=RedisDBConfig.HOST, port=RedisDBConfig.PORT)
        p = r.pipeline()
        for i in range(len(userid)):
            if i%1000 == 0:
                print i
                p.execute()
            p.hset(self.userPopular, userid[i], data[i])
        p.execute()
        logging.debug('saveToRedis')

    def popular(self):
        """
        计算逻辑
        l[1]+l[1]/l[0]+l[2]+l[2]*l[3]
        """
        raw = self.loadData()
        rawArray  = np.array(raw, dtype=float)
        
        userid = [i[0] for i in raw]
        data = rawArray[:,1:]

        row = data.shape[0]

        dataTemp = (data[:,1]).reshape(row,1)
        dataTemp = np.hstack((dataTemp, (data[:,1]/data[:,0]).reshape(row,1)))
        
        dataTemp = np.hstack((dataTemp, (data[:,2]).reshape(row,1)))
        
        dataTemp = np.hstack((dataTemp, (data[:,2]*(data[:,3]-20000)).reshape(row,1)))


        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        data_minmax = min_max_scaler.fit_transform(dataTemp)

        resultData = list(data_minmax[:,0] + data_minmax[:,1] + data_minmax[:,2] + data_minmax[:,3])
        self.saveToRedis(userid, resultData)
        
if __name__ == '__main__':
    pop = Popular()
    pop.popular()

