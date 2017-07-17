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
import urllib2
import urllib


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
        sql = "select t1.userid,case when like_score>1 then 1 else like_score end as like_score,liked_score,conn_score,report_score, \
case when block_score>1 then 1 else block_score end as block_score,hangup_score,pay_score from \
(select userid, \
max(like_count+2)*1.0/(max(conn_count)+5) as like_score,\
(log(max(liked_count)+1))*max(liked_count)*1.0/(max(like_count)+max(liked_count)+2.0)/(max(conn_count)+5) as liked_score, \
log(max(conn_count)+1)*(max(conn_avg)-10000) as conn_score, \
max(reportedcount)*1.0/(max(conn_count)+5) as report_score, \
max(blockcount)*1.0/(max(conn_count)+5) as block_score, \
(max(LIVING_HANGUP_10)+max(LIVING_USER_HANGUP_10))*1.0/(max(LIVING_USER_HANGUP)+max(LIVING_HANGUP)+4.0) as hangup_score \
from \
  (select userid,reportedcount,blockcount,like_count,liked_count,conn_count,conn_avg \
  from meshmatch_user_ext where conn_avg>0) m  \
  left join  \
  (select n.leftid,LIVING_HANGUP,LIVING_USER_HANGUP,LIVING_HANGUP_10,LIVING_USER_HANGUP_10 from  \
  (select leftid,sum(case when eventtype=0 then cnt else 0 end) as LIVING_HANGUP,sum(case when eventtype=1 then cnt else 0 end) as LIVING_USER_HANGUP \
    from (select leftid,eventtype,count(*) as cnt  \
      from (select distinct leftid,rightid,eventtype  \
        from (select leftid,rightid,eventtype  \
          from (select fromuserid as leftid,touserid as rightid,case when eventtype='LIVING_USER_HANGUP' then 0 else 1 end as eventtype  \
            from meshmatch_event_prod where eventtype in ('LIVING_HANGUP','LIVING_USER_HANGUP')) a  \
            union all  \
                select touserid as leftid,fromuserid as rightid,case when eventtype='LIVING_HANGUP' then 0 else 1 end as eventtype  \
                  from meshmatch_event_prod where eventtype in ('LIVING_HANGUP','LIVING_USER_HANGUP')) b) c  \
                  group by leftid,eventtype)d group by leftid) n  \
  left join  \
  (select leftid,sum(case when eventtype=0 then cnt else 0 end) as LIVING_HANGUP_10,sum(case when eventtype=1 then cnt else 0 end) as LIVING_USER_HANGUP_10 \
    from (select leftid,eventtype,count(*) as cnt  \
      from (select distinct leftid,rightid,eventtype  \
        from (select leftid,rightid,eventtype  \
          from (select fromuserid as leftid,touserid as rightid,case when eventtype='LIVING_USER_HANGUP' then 0 else 1 end as eventtype  \
            from meshmatch_event_prod where eventtype in ('LIVING_HANGUP','LIVING_USER_HANGUP') and duration<10000) a  \
            union all  \
                select touserid as leftid,fromuserid as rightid,case when eventtype='LIVING_HANGUP' then 0 else 1 end as eventtype  \
                  from meshmatch_event_prod where eventtype in ('LIVING_HANGUP','LIVING_USER_HANGUP') and duration<10000) b) c  \
                  group by leftid,eventtype)d group by leftid) p \
                  on n.leftid=p.leftid) q    \
   on m.userid=q.leftid group by userid) t1 left join (select distinct userid,1 as pay_score from meshmatch_payrecord_prod) t2 on t1.userid=t2.userid;"

        sqlResult = self.redshift.getAll(sql)
        userid = [i[0] for i in sqlResult]
        rawArray  = np.array(sqlResult, dtype=float)
        data = rawArray[:,1:]
        df = pd.DataFrame(data, index=userid)
        #df2 = df.fillna(0)

        logging.debug('loadData')
        return df

    def loadSensorsData(self, url, val):
        data = urllib.urlencode(val) 
        request = urllib2.Request(url,data)
        response = urllib2.urlopen(request)
        sensor = response.read()
        row = sensor.split('\n')
        colKeys = row[0].split('\t')
        midL = []
        cntL = []
        length = len(sensor.split('\n'))
        for i in range(1,length-1):
            col = row[i].split('\t')
            if len(col) == 2:
                midL.append(col[0])
                cntL.append(col[1])


        raw_data = {'cntL': cntL}
        df = pd.DataFrame(raw_data, index=midL, columns=['cntL'])
        return df

    def saveToRedis(self, userid, data, minResult, span):
        r = redis.Redis(host=RedisDBConfig.HOST, port=RedisDBConfig.PORT)
        p = r.pipeline()
        for i in range(len(userid)):
            if i%1000 == 0:
                print i
                p.execute()
            p.hset(self.userPopular, userid[i], (data[i]-minResult)/span*5.0)
        p.execute()
        logging.debug('saveToRedis')

    def popular(self):
        """
        计算逻辑
        """
        url = 'http://sensors.sta.holaverse.com:8007/api/sql/query?token=5a14139d09916d172f9e99375ebdf78c0dc01bf14a8a2fbe55eeed45a9521bb1&project=tiki'
        val = {"q":"select mid,(sum(case when int_val=1 then cnt else 0.0 end)+sum(case when int_val=3 then cnt else 0 end)+4.0)/(sum(cnt)+8.0) from (select mid,int_val,count(int_val) as cnt from events where event='M123' group by mid,int_val) a group by mid"}
        sensor = self.loadSensorsData(url, val)
        redShiftData = self.loadData()
        result = pd.concat([redShiftData, sensor], axis=1)
        df1 = result.fillna(0)

        userid = list(df1.index)
        data = np.array(df1,dtype=float)
        row = data.shape[0]
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        data_minmax = min_max_scaler.fit_transform(data)

        #like_score,liked_score,conn_score,report_score,block_score,hangup_score,pay_score,异常断开
        resultData = list(data_minmax[:,0] + 3*data_minmax[:,1] + data_minmax[:,2] - 2*data_minmax[:,3] - 2*data_minmax[:,4] - data_minmax[:,5] + 2*data_minmax[:,6] - 2*data_minmax[:,7])
        
        maxResult = max(resultData)
        minResult = min(resultData)
        span = maxResult-minResult
        """
        测试
        u = np.array(userid)

        pos=np.where(u=='42462576')
        resultData[14270]
        data_minmax[14270]

        pos2 = np.where(resultData>resultData[14270])
        len(u[pos2])


        pos=np.where(u=='77534097') 
        resultData[29551]
        pos2 = np.where(resultData>resultData[29551])
        u[pos2]

        pos=np.where(u=='22843255') 
        resultData[5706]
        pos2 = np.where(resultData>resultData[5706])
        len(u[pos2])
        
        """
    
        self.saveToRedis(userid, resultData, minResult, span)
        
def checkFun(pNum=1):
    """
    #执行检查当前进程数目
    """
    #cmd = "ps aux | grep 'python main_zhx.py' | grep -v 'grep' | wc -l"
    cmd = "ps aux | grep 'popular_recommend.py' | grep -v 'grep' | wc -l"
    num = int(os.popen(cmd).readlines()[0])
    if (num > pNum):
        sys.exit('Process has running!')

if __name__ == '__main__':
    checkFun()

    pop = Popular()
    pop.popular()

