#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pandas as pd
import numpy as np 
from copy import deepcopy
import MySQLdb
import datetime
import time

from multiprocessing import Queue, Process, Lock

from math import isnan
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima_model import ARIMA

from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

from sklearn import preprocessing
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

#使用装饰器(decorator),  单例 
def singleton(cls, *args, **kw):  
    instances = {}  
    def _singleton():  
        if cls not in instances:  
            instances[cls] = cls(*args, **kw)
        return instances[cls]
    return _singleton

PROCESS_NUM = 20

def inputQ(q, data):
    q.put(data)

def outputQ(q, lock):
    lock.acquire() 
    try:
        return q.get(timeout=10)
    except Exception, e:
        #model.Log.info('Queue time out %s' % str(e))
        return None
    finally:
        lock.release()

class baseModel(object):
    """docstring for baseModel
    基类初始化数据库链接
    """
    def __init__(self):
        super(baseModel, self).__init__()
        self.db = None
        self.host = "test-dataverse-web.c0poh9vgjxya.rds.cn-north-1.amazonaws.com.cn"
        self.username = "datauser"
        self.passwd = "ebOUeWSin3hAAKKD"
        self.database = "dataverse"

    def conn(self):
        """
        mysql连接
        """
        try:
            if self.db == None:
                self.db = MySQLdb.connect(self.host,self.username,self.passwd,self.database)
            return self.db
        except Exception, e:
            print "open db error"
    
    def close(self):
        """
        关闭数据库连接
        """
        if self.db is None:
            return True
        try:
            self.db.close()
        except Exception, e:
            print "close db error"

#公共方法
def getDic(data, keyList, valueList):
    dataDic={} #是一个规划数据的字典，字典key有产品号和渠道号联合组成
    for i in range(len(data)):
        key=""
        for j in keyList:
            key = key + str(data[i][j]) + '#'
        #key = key.strip("#")
        value = []
        for v in valueList:
            #print str(data[i][v])
            value.extend([str(data[i][v])])
        #保存数据到字典
        if dataDic.has_key(key):
            dataDic[key].extend([value])
        else:
            dataDic[key]=[value]

    return dataDic

def getPlusDay(day, plus):
    date = datetime.datetime.strptime(str(day),"%Y%m%d") + datetime.timedelta(days=plus)
    string = date.strftime("%Y%m%d")
    return string

def formatTime(strs):
    strs = str(strs)
    return strs[0:4] + '-' + strs[4:6] + '-' + strs[6:8]

def dataDiff(last, begin):
    date1=datetime.datetime.strptime(str(begin),"%Y%m%d")
    date2=datetime.datetime.strptime(str(last),"%Y%m%d")
    return (date2 - date1).days + 1


class ARIMAModel(baseModel):
    """
    ARIMA操作model类
    """
    def __init__(self, tablename, whereSql, predictDays, beginDay, endDay, colname):
        """
        参数解释：
        tablename:预测的表名
        whereSql:根据预测key构造的where条件
        predictDays:待预测的天数
        beginDay:获取已知数据的开始日期
        endDay： 获取已知数据的结束日期
        colname:保存列名
        """
        baseModel.__init__(self)
        self.table = tablename
        self.today = datetime.date.today().strftime("%Y%m%d")
        self.forecastnum = predictDays
        self.tsColName = ['cnt']
        self.colname = colname
        #self.daysAgo3 = getPlusDay(self.today, -4)

        if self.table == "ROI_Retained_Predict_GroupBy_c":
            self.keyList = [0,2,3,4,5]
            self.valueList = [1,6]
            if self.colname == "retain":
                self.insertSql = "INSERT INTO ROI_Retained_Predict_GroupBy_c (`date`, retain_day, pdtid, country_iso,channel_category, pid, retain_predict) VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE retain_predict=%s;"
                self.selectSql = "select date,retain_day,pdtid,country_iso,channel_category,pid,if(1/retain,retain,retain_predict)  as retain from ROI_Retained_Predict_GroupBy_c where " + whereSql + " and retain_day<" + endDay + " and retain_day>="+ beginDay +";"
            elif self.colname == "spec_retain":
                self.insertSql = "INSERT INTO ROI_Retained_Predict_GroupBy_c (`date`, retain_day, pdtid, country_iso,channel_category, pid, spec_retain_predict) VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE spec_retain_predict=%s;"
                self.selectSql = "select date,retain_day,pdtid,country_iso,channel_category,pid,if(1/spec_retain,spec_retain,spec_retain_predict)  as spec_retain from ROI_Retained_Predict_GroupBy_c where " + whereSql + " and retain_day<" + endDay + " and retain_day>="+ beginDay +";"
            
        elif self.table == "ROI_Purchase_Predict_GroupBy_c":
            self.keyList = [0,2,3,4,5]
            self.valueList = [1,6]
            self.insertSql = "INSERT INTO ROI_Purchase_Predict_GroupBy_c (`date`,revenue_day,pdtid,country_iso,channel_category,pid,purchase_predict) VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE purchase_predict=%s;"
            self.selectSql = "select date,revenue_day,pdtid,country_iso,channel_category,pid,if(1/purchase,purchase,purchase_predict)  as purchase from ROI_Purchase_Predict_GroupBy_c where " + whereSql + " and revenue_day<" + endDay + " and revenue_day>="+ beginDay +";"
            
        elif self.table == "ROI_News_Predict_GroupBy_c":
            self.keyList = [0,2,3,4,5,6,7]
            self.valueList = [1,8]

            if self.colname == "imp_num":
                self.insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_num_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE imp_num_predict=%s;"

                self.selectSql = "select `date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,if(1/imp_num,imp_num,imp_num_predict) as imp_num from ROI_News_Predict_GroupBy_c where " + whereSql + " and revenue_date<" + endDay + " and revenue_date>="+ beginDay +";"
            elif self.colname == "click_num":
                self.insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,click_num_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE click_num_predict=%s;"
                self.selectSql = "select `date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,if(1/click_num,click_num,click_num_predict) as click_num from ROI_News_Predict_GroupBy_c where " + whereSql + " and revenue_date<" + endDay + " and revenue_date>="+ beginDay +";"
            elif self.colname == "imp_revenue":
                self.insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_revenue_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE imp_revenue_predict=%s;"
                self.selectSql = "select `date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,if(1/imp_revenue,imp_revenue,imp_revenue_predict) as imp_revenue from ROI_News_Predict_GroupBy_c where " + whereSql + " and revenue_date<" + endDay + " and revenue_date>="+ beginDay +";"
            elif self.colname == "click_revenue":
                self.insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,click_revenue_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE click_revenue_predict=%s;"
                self.selectSql = "select `date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,if(1/click_revenue,click_revenue,click_revenue_predict) as click_revenue from ROI_News_Predict_GroupBy_c where " + whereSql + " and revenue_date<" + endDay + " and revenue_date>="+ beginDay +";"
            else:
                print "Please write right column name!"
                exit()


    def definePQ(self, data):
        c = 1
        #一般阶数不超过length/10
        pmax = int(len(data)/10)
        #一般阶数不超过length/10
        qmax = int(len(data)/10)
        
        data=data.astype('float64')
        #bic矩阵
        bic_matrix = [] 
        for p in range(pmax+1):
            tmp = [] #tmp保存的是计算的bic值
            for q in range(qmax+1):#存在部分报错，所以用try来跳过报错。
                try: 
                    r = ARIMA(data, (p,1,q)).fit().bic
                    if isnan(r):
                        tmp.append(p+q+1000000000)
                    else:
                        tmp.append(r)
                except:
                    #如果bic值为None，则保存一个最大的整数来代表bic
                    tmp.append(p+q+1000000000)
            #print tmp
            bic_matrix.append(tmp)
        #从中可以找出最小值
        bic_matrix = pd.DataFrame(bic_matrix) 
        #先用stack展平，然后用idxmin找出最小值位置。
        p,q = bic_matrix.stack().idxmin() 
        bicMin = bic_matrix.stack()[p,q]
        if bicMin >= 1000000000:
            return 0, 0, 0 
        print(u'BIC最小的p值c值q值为：%s、%s、%s' %(p,c,q)) 
        return p, c, q

    #建立ARIMA(p, c, q)模型
    def buildARIMA(self, data, start, end):
        p,c,q = self.definePQ(data)
        print data
        model = ARIMA(data, (p,c,q)).fit()
        
        if c == 0:
            result = model.predict(start=start, end=end)
        else:
            result = model.predict(start=start, end=end, typ='levels')
        result = np.exp(result)
        return result
    
    def getIndexCnt(self, value, td, length):
        #print td
        indexs = []
        cnt = []
        if td == length:
            for j in range(td):
                indexs.append(value[j][0])
                cnt.append(float(value[j][1]))
        #处理连续日期，中间有数据缺失的情况
        else:
            j = 0
            begintime = str(value[0][0])
            for i in range(td):
                nowtime = getPlusDay(begintime ,i)
                if nowtime == str(value[j][0]):
                    indexs.append(value[j][0])
                    cnt.append(float(value[j][1]))
                    j+=1
                else:
                    #如果数据连续缺失超过一天，则不预测。
                    df=dataDiff(value[j][0],value[j-1][0])-1
                    if df > 2:
                        return [],[]
                    indexs.append(long(nowtime))
                    print value[j][1]
                    print value[j-1][1]
                    print value
                    cnt.append((float(float(value[j][1])+float(value[j-1][1]))/2.0))
    
        return indexs, cnt

    def predict(self):
        db = baseModel().conn()
        cursor = db.cursor()
        cursor.execute(self.selectSql)
        data = cursor.fetchall()
        
        dic = getDic(data, self.keyList, self.valueList)
        #print dic
        for key in dic:
            length = len(dic[key])
            keys = key.split("#")
            
            value = dic[key]
            value.sort()
            #计算开始时间和结束时间的相隔天数
            last = value[-1][0]
            begin = value[0][0]
            td = dataDiff(last, begin)
            
            #获取索引和计数值
            indexs, cnt = self.getIndexCnt(value, td, length)
            
            if len(indexs) == 0:
                continue
        
            ts = pd.DataFrame(cnt, index=indexs, columns=self.tsColName)
            ts['day'] = ts.index
            ts.index = pd.to_datetime(ts['day'].apply(formatTime))
            
            data = np.log(ts['cnt'])
            
            if length>10:
                resultValues = []
                datats = data[2:]
                #print datats
                newDate = datats.index[-1] + relativedelta(days=1)
                endData = datats.index[-1] + relativedelta(days=self.forecastnum)
                newDateStr = newDate.strftime("%Y-%m-%d")
                endDataStr = endData.strftime("%Y-%m-%d")
                #print datats #a=ARIMA(datats, (0,0,0)).fit() #print a.predict(start='2016-11-03',end='2016-11-03')
                result = self.buildARIMA(datats,newDateStr,endDataStr)
                #print result
                for i in range(self.forecastnum):
                    dateStr = (newDate + relativedelta(days=i)).strftime("%Y-%m-%d").replace("-","")
                    if self.table == "ROI_Retained_Predict_GroupBy_c":
                        resultValues = [keys[0], dateStr, keys[1], keys[2], keys[3], keys[4], result[i], result[i]]
                    elif self.table == "ROI_Purchase_Predict_GroupBy_c":
                        resultValues = [keys[0], dateStr, keys[1], keys[2], keys[3], keys[4], result[i], result[i]]
                    elif self.table == "ROI_install_Predict_GroupBy_c":
                        resultValues = [dateStr, keys[0], keys[1], keys[2], keys[3], result[i], result[i]]
                    elif self.table == "ROI_News_Predict_GroupBy_c":
                        resultValues = [keys[0], dateStr, keys[1], keys[2], keys[3], keys[4], keys[5], keys[6], result[i], result[i]]
                    #print resultValues
                    #print insertSql
                    cursor.execute(self.insertSql, resultValues)
                cursor.execute("commit;")
        
        cursor.execute("commit;")
        # 关闭数据库连接
        baseModel().close()


class RegModel(baseModel):
    """
    线性回归model类
    """
    def __init__(self, tablename, predictDays, colname, queue):
        """
        参数解释：
        tablename:预测的表名
        predictDays:待预测的天数
        """
        baseModel.__init__(self)

        self.q = queue

        self.today = datetime.date.today().strftime("%Y%m%d")
        self.predictDays = predictDays
        self.table = tablename
        self.colname = colname
        self.yesterday = getPlusDay(self.today, -1)
        self.daysAgo3 = getPlusDay(self.today, -3)
        self.daysAgo60 = getPlusDay(self.today, -60)
        self.daysAgo90 = getPlusDay(self.today, -90)
        self.threshold = 0.6

        if self.table == "ROI_Retained_Predict_GroupBy_c":
            self.pdtidSql = "select distinct pdtid from ROI_Retained  where pdtid<>'' and country_iso<>'' and channel_category<>'' and pid<>'' and retain_day>="+ self.daysAgo3 +" and date>="+self.daysAgo90+";"
            self.exeSql = "INSERT INTO ROI_Retained_Predict_GroupBy_c (`date`, retain_day, pdtid, country_iso, channel_category, pid, retain, spec_retain) select date,retain_day,pdtid,ifnull(country_iso,'') as country_iso,ifnull(channel_category,'') as channel_category,ifnull(pid,'') as pid,retain, spec_retain from (select date,retain_day,pdtid,country_iso,channel_category,pid,sum(retain) as retain,sum(spec_retain) as spec_retain from (select date,retain_day,pdtid,country_iso,channel_category,pid,sum(retain) as retain,sum(spec_retain) as spec_retain from ROI_Retained  where pdtid=%s and country_iso<>'' and channel_category<>'' and pid<>'' and retain_day>="+ self.daysAgo3 +" and date>="+self.daysAgo90+" group by date,retain_day,pdtid,country_iso,channel_category,pid) t1 group by date,retain_day,pdtid,country_iso,channel_category,pid WITH ROLLUP union select date,retain_day,pdtid,country_iso,channel_category,pid,sum(retain) as retain,sum(spec_retain) as spec_retain from (select date,retain_day,pdtid,country_iso,channel_category,pid,sum(retain) as retain,sum(spec_retain) as spec_retain from ROI_Retained  where pdtid=%s and country_iso<>'' and channel_category<>'' and pid<>'' and retain_day>="+ self.daysAgo3 +" and date>="+self.daysAgo90+" group by date,retain_day,pdtid,country_iso,channel_category,pid) t2 group by date,retain_day,pdtid,channel_category,pid,country_iso WITH ROLLUP) t3 where pdtid is not NULL on DUPLICATE KEY UPDATE retain=t3.retain,spec_retain=t3.spec_retain;"
            self.keyList = [1,2,3,4]
            self.valueList = [5,7]
        
            self.predictKeyList = [2,3,4,5]
            self.predictValueList = [0,6,7]
        
            self.keyName = ['pdtid', 'country_iso', 'channel_category', 'pid']

            if self.colname == "retain":
                self.insertSql = "INSERT INTO ROI_Retained_Predict_GroupBy_c (`date`, retain_day, pdtid, country_iso,channel_category, pid,retain_predict) VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE retain_predict=%s;"
                #获取经过处理的mysql数据
                self.selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,datediff(str_to_date(a.retain_day,'%%Y%%m%%d'),str_to_date(b.retain_day,'%%Y%%m%%d')) as day_diff,b.retain_day as retain_day_min,log(b.retain/a.retain) as log_retain_rate,b.retain from (select * from ROI_Retained_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and retain_day<" + self.today + "  and retain>0) a,(select date,min(retain_day) as retain_day,pdtid, country_iso,channel_category,pid,retain as retain from ROI_Retained_Predict_GroupBy_c where pdtid=%s and retain_day<" + self.today + " and retain>0 and date>=" + self.daysAgo60 + " group by date,pdtid,country_iso,channel_category,pid) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid) d where day_diff>0 and log_retain_rate is not null;"
                #调试sql
                #self.selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,datediff(str_to_date(a.retain_day,'%%Y%%m%%d'),str_to_date(b.retain_day,'%%Y%%m%%d')) as day_diff,b.retain_day as retain_day_min,log(b.retain/a.retain) as log_retain_rate,b.retain from (select * from ROI_Retained_Predict_GroupBy_c where date>=" + self.daysAgo60 + " and retain_day<" + self.today + " and pdtid='600027' and country_iso='US'  and channel_category='' and retain>0) a,(select date,min(retain_day) as retain_day,pdtid, country_iso,channel_category,pid,retain as retain from ROI_Retained_Predict_GroupBy_c where retain_day<" + self.today + " and retain>0 and pdtid='600027' and country_iso='US'  and channel_category='' and date>=" + self.daysAgo60 + " group by date,pdtid,country_iso,channel_category,pid) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid) d where day_diff>0 and log_retain_rate is not null;"
        
                self.predictSelectSql = "select b.date,b.retain_day,b.pdtid,b.country_iso,b.channel_category,b.pid,b.retain,datediff(str_to_date(b.retain_day,'%%Y%%m%%d'),str_to_date(b.date,'%%Y%%m%%d')) as day_diff from (select date,max(retain_day) as retain_day,pdtid,country_iso,channel_category,pid from ROI_Retained_Predict_GroupBy_c where pdtid=%s and retain_day<" + self.today + " and retain>0 and date>=" + self.daysAgo60 + " group by date,pdtid,country_iso,channel_category,pid) a,ROI_Retained_Predict_GroupBy_c b where b.pdtid=%s and a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.retain_day=b.retain_day;"
            elif self.colname == "spec_retain":
                self.insertSql = "INSERT INTO ROI_Retained_Predict_GroupBy_c (`date`, retain_day, pdtid, country_iso,channel_category, pid,spec_retain_predict) VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE spec_retain_predict=%s;"
                #获取经过处理的mysql数据
                self.selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,datediff(str_to_date(a.retain_day,'%%Y%%m%%d'),str_to_date(b.retain_day,'%%Y%%m%%d')) as day_diff,b.retain_day as retain_day_min,log(b.spec_retain/a.spec_retain) as log_retain_rate,b.spec_retain from (select * from ROI_Retained_Predict_GroupBy_c where pdtid=%s and retain_day<" + self.today + " and spec_retain>0 and date>=" + self.daysAgo60 + ") a,(select date,min(retain_day) as retain_day,pdtid,country_iso,channel_category,pid,spec_retain as spec_retain from ROI_Retained_Predict_GroupBy_c where pdtid=%s and retain_day<" + self.today + " and spec_retain>0 and date>=" + self.daysAgo60 + " group by date,pdtid,country_iso,channel_category,pid) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid) d where day_diff>0 and log_retain_rate is not null;"

                self.predictSelectSql = "select b.date,b.retain_day,b.pdtid,b.country_iso,b.channel_category,b.pid,b.spec_retain,datediff(str_to_date(b.retain_day,'%%Y%%m%%d'),str_to_date(b.date,'%%Y%%m%%d')) as day_diff from (select date,max(retain_day) as retain_day,pdtid,country_iso,channel_category,pid from ROI_Retained_Predict_GroupBy_c where pdtid=%s and retain_day<" + self.today + " and spec_retain>0 and date>=" + self.daysAgo60 + " group by date,pdtid,country_iso,channel_category,pid) a,ROI_Retained_Predict_GroupBy_c b where b.pdtid=%s and a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.retain_day=b.retain_day;"
        elif self.table == "ROI_Purchase_Predict_GroupBy_c":
            self.threshold = 0.45
            self.pdtidSql = "select distinct pdtid from ROI_Purchase where pdtid<>'' and country_iso<>'' and channel_category<>'' and pid<>'' and revenue_day>="+ self.daysAgo3 +";"
            self.exeSql = "INSERT INTO ROI_Purchase_Predict_GroupBy_c (`date`, revenue_day, pdtid, country_iso, channel_category, pid, purchase) select date,revenue_day,pdtid,ifnull(country_iso,'') as country_iso,ifnull(channel_category,'') as channel_category,ifnull(pid,'') as pid,purchase from (select date,revenue_day,pdtid,country_iso,channel_category,pid,sum(purchase) as purchase from (select date,revenue_day,pdtid,country_iso,channel_category,pid,sum(purchase) as purchase from ROI_Purchase  where pdtid=%s and country_iso<>'' and channel_category<>'' and pid<>'' and revenue_day>="+ self.daysAgo3 + " group by date,revenue_day,pdtid,country_iso,channel_category,pid) t1 group by date,revenue_day,pdtid,country_iso,channel_category,pid WITH ROLLUP union select date,revenue_day,pdtid,country_iso,channel_category,pid,sum(purchase) as purchase from (select date,revenue_day,pdtid,country_iso,channel_category,pid,sum(purchase) as purchase from ROI_Purchase  where pdtid=%s and country_iso<>'' and channel_category<>'' and pid<>'' and revenue_day>="+ self.daysAgo3 +" group by date,revenue_day,pdtid,country_iso,channel_category,pid) t2 group by date,revenue_day,pdtid,channel_category,pid,country_iso WITH ROLLUP) t3 where pdtid is not NULL and date<=revenue_day on DUPLICATE KEY UPDATE purchase=t3.purchase;;"
            self.keyList = [1,2,3,4]
            self.valueList = [5,7]
        
            self.predictKeyList = [2,3,4,5]
            self.predictValueList = [0,6,7]

            self.keyName = ['pdtid', 'country_iso', 'channel_category', 'pid']
        
            self.insertSql = "INSERT INTO ROI_Purchase_Predict_GroupBy_c (`date`,revenue_day,pdtid,country_iso,channel_category,pid,purchase_predict) VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE purchase_predict=%s;"
            self.selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,datediff(str_to_date(a.revenue_day,'%%Y%%m%%d'),str_to_date(b.revenue_day,'%%Y%%m%%d')) as day_diff,b.revenue_day as purchase_day_min,log(b.purchase/a.purchase) as log_purchase_rate,b.purchase from (select * from ROI_Purchase_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and revenue_day<" + self.today + " and purchase>0) a,(select date,min(revenue_day) as revenue_day,pdtid,country_iso,channel_category,pid,purchase from ROI_Purchase_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and revenue_day<" + self.today + " and purchase>0 group by date,pdtid,country_iso,channel_category,pid) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid) d where day_diff>0 and log_purchase_rate is not null;"
            #测试
            #self.selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,datediff(str_to_date(a.revenue_day,'%%Y%%m%%d'),str_to_date(b.revenue_day,'%%Y%%m%%d')) as day_diff,b.revenue_day as purchase_day_min,log(b.purchase/a.purchase) as log_purchase_rate,b.purchase from (select * from ROI_Purchase_Predict_GroupBy_c where date>=" + self.daysAgo60 + " and revenue_day<" + self.today + " and purchase>0 and pdtid=600020 and country_iso='' and pid='Organic') a,(select date,min(revenue_day) as revenue_day,pdtid,country_iso,channel_category,pid,purchase from ROI_Purchase_Predict_GroupBy_c where date>=" + self.daysAgo60 + " and revenue_day<" + self.today + " and purchase>0 and pdtid=600020 and country_iso='' and pid='Organic' group by date,pdtid,country_iso,channel_category,pid) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid) d where day_diff>0 and log_purchase_rate is not null;"
            self.predictSelectSql = "select b.date,b.revenue_day,b.pdtid,b.country_iso,b.channel_category,b.pid,b.purchase,datediff(str_to_date(b.revenue_day,'%%Y%%m%%d'),str_to_date(b.date,'%%Y%%m%%d')) as day_diff from (select date,max(revenue_day) as revenue_day,pdtid,country_iso,channel_category,pid from ROI_Purchase_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and revenue_day<" + self.today + " and purchase>0 group by date,pdtid,country_iso,channel_category,pid) a,ROI_Purchase_Predict_GroupBy_c b where b.pdtid=%s and a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.revenue_day=b.revenue_day ;"
        elif self.table == "ROI_News_Predict_GroupBy_c":
            self.keyList = [1,2,3,4,5,6]
            self.valueList = [7,9]

            self.predictKeyList = [2,3,4,5,6,7]
            self.predictValueList = [0,8,9]

            self.keyName = ['pdtid', 'country_iso', 'channel_category', 'pid', 'ad_type_id', 'ad_type']
            self.pdtidSql = "select distinct pdtid from ROI_News where pdtid<>'' and country_iso<>'' and channel_category<>'' and pid<>'' and revenue_date>="+ self.daysAgo3 + ";"
            self.exeSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_num,click_num,imp_revenue,click_revenue) \
                select date,revenue_date,pdtid,ifnull(country_iso,'') as country_iso,ifnull(channel_category,'') as channel_category,ifnull(pid,'') as pid,ifnull(ad_type_id,'') as ad_type_id,ifnull(ad_type,'') as ad_type,imp_num,click_num,imp_revenue,click_revenue from \
                    (select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,sum(imp_num) as imp_num,sum(click_num) as click_num,sum(imp_revenue) as imp_revenue,sum(click_revenue) as click_revenue from (select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,sum(imp_num) as imp_num,sum(click_num) as click_num,sum(imp_revenue) as imp_revenue,sum(click_revenue) as click_revenue from ROI_News where pdtid=%s and country_iso<>'' and channel_category<>'' and pid<>'' and ad_type_id<>'' and ad_type<>'' and revenue_date>="+ self.daysAgo3 + " group by date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) t1 group by date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type WITH ROLLUP \
                    union \
                    select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,sum(imp_num) as imp_num,sum(click_num) as click_num,sum(imp_revenue) as imp_revenue,sum(click_revenue) as click_revenue from (select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,sum(imp_num) as imp_num,sum(click_num) as click_num,sum(imp_revenue) as imp_revenue,sum(click_revenue) as click_revenue from ROI_News where pdtid=%s and country_iso<>'' and channel_category<>'' and pid<>'' and ad_type_id<>'' and ad_type<>'' and revenue_date>="+ self.daysAgo3 + " group by date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) t2 group by date,revenue_date,pdtid,country_iso,ad_type_id,ad_type,channel_category,pid WITH ROLLUP \
                    union \
                    select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,sum(imp_num) as imp_num,sum(click_num) as click_num,sum(imp_revenue) as imp_revenue,sum(click_revenue) as click_revenue from (select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,sum(imp_num) as imp_num,sum(click_num) as click_num,sum(imp_revenue) as imp_revenue,sum(click_revenue) as click_revenue from ROI_News where pdtid=%s and country_iso<>'' and channel_category<>'' and pid<>'' and ad_type_id<>'' and ad_type<>'' and revenue_date>="+ self.daysAgo3 + " group by date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) t3 group by date,revenue_date,pdtid,channel_category,pid,ad_type_id,ad_type,country_iso WITH ROLLUP \
                    union \
                    select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,sum(imp_num) as imp_num,sum(click_num) as click_num,sum(imp_revenue) as imp_revenue,sum(click_revenue) as click_revenue  from (select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,sum(imp_num) as imp_num,sum(click_num) as click_num,sum(imp_revenue) as imp_revenue,sum(click_revenue) as click_revenue from ROI_News where pdtid=%s and country_iso<>'' and channel_category<>'' and pid<>'' and ad_type_id<>'' and ad_type<>'' and revenue_date>="+ self.daysAgo3 + " group by date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) t4 group by date,revenue_date,pdtid,channel_category,pid,country_iso,ad_type_id,ad_type WITH ROLLUP \
                    union \
                    select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,sum(imp_num) as imp_num,sum(click_num) as click_num,sum(imp_revenue) as imp_revenue,sum(click_revenue) as click_revenue  from (select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,sum(imp_num) as imp_num,sum(click_num) as click_num,sum(imp_revenue) as imp_revenue,sum(click_revenue) as click_revenue from ROI_News where pdtid=%s and country_iso<>'' and channel_category<>'' and pid<>'' and ad_type_id<>'' and ad_type<>'' and revenue_date>="+ self.daysAgo3 + " group by date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) t5 group by date,revenue_date,pdtid,ad_type_id,ad_type,country_iso,channel_category,pid WITH ROLLUP \
                    union \
                    select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,sum(imp_num) as imp_num,sum(click_num) as click_num,sum(imp_revenue) as imp_revenue,sum(click_revenue) as click_revenue  from (select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,sum(imp_num) as imp_num,sum(click_num) as click_num,sum(imp_revenue) as imp_revenue,sum(click_revenue) as click_revenue from ROI_News where pdtid=%s and country_iso<>'' and channel_category<>'' and pid<>'' and ad_type_id<>'' and ad_type<>'' and revenue_date>="+ self.daysAgo3 + " group by date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) t6 group by date,revenue_date,pdtid,ad_type_id,channel_category,pid,ad_type,country_iso WITH ROLLUP) t where pdtid is not NULL and date<=revenue_date on DUPLICATE KEY UPDATE imp_num=t.imp_num,click_num=t.click_num,imp_revenue=t.imp_revenue,click_revenue=t.click_revenue;"

            if self.colname == "imp_num":
                self.insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_num_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE imp_num_predict=%s;"
                self.selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,a.ad_type_id,a.ad_type,datediff(str_to_date(a.revenue_date,'%%Y%%m%%d'),str_to_date(b.revenue_date,'%%Y%%m%%d')) as day_diff,b.revenue_date as purchase_day_min,log(b.imp_num/a.imp_num) as log_purchase_rate,b.imp_num from (select * from ROI_News_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and revenue_date<" + self.today + " and imp_num>0) a,(select date,min(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_num from ROI_News_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and revenue_date<" + self.today + " and imp_num>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type) d where day_diff>0 and log_purchase_rate is not null;"
                #测试
                #self.selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,a.ad_type_id,a.ad_type,datediff(str_to_date(a.revenue_date,'%%Y%%m%%d'),str_to_date(b.revenue_date,'%%Y%%m%%d')) as day_diff,b.revenue_date as purchase_day_min,log(b.imp_num/a.imp_num) as log_purchase_rate,b.imp_num from (select * from ROI_News_Predict_GroupBy_c where revenue_date<" + self.today + " and pdtid=600001 and country_iso='ES' and channel_category='Organic' and pid='Organic' and ad_type_id='admob' and ad_type='video' and imp_num>0) a,(select date,min(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_num from ROI_News_Predict_GroupBy_c where revenue_date<" + self.today + " and imp_num>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type) d where day_diff>0 and log_purchase_rate is not null;"
                self.predictSelectSql = "select b.date,b.revenue_date,b.pdtid,b.country_iso,b.channel_category,b.pid,b.ad_type_id,b.ad_type,b.imp_num,datediff(str_to_date(b.revenue_date,'%%Y%%m%%d'),str_to_date(b.date,'%%Y%%m%%d')) as day_diff from (select date,max(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type from ROI_News_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and revenue_date<" + self.today + " and imp_num>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) a,ROI_News_Predict_GroupBy_c b where b.pdtid=%s and a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.revenue_date=b.revenue_date and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type;"
            elif self.colname == "click_num":
            
                self.insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,click_num_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE click_num_predict=%s;"
                self.selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,a.ad_type_id,a.ad_type,datediff(str_to_date(a.revenue_date,'%%Y%%m%%d'),str_to_date(b.revenue_date,'%%Y%%m%%d')) as day_diff,b.revenue_date as purchase_day_min,log(b.click_num/a.click_num) as log_purchase_rate,b.click_num from (select * from ROI_News_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and revenue_date<" + self.today + " and click_num>0) a,(select date,min(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,click_num from ROI_News_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and revenue_date<" + self.today + " and click_num>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type) d where day_diff>0 and log_purchase_rate is not null;"
                self.predictSelectSql = "select b.date,b.revenue_date,b.pdtid,b.country_iso,b.channel_category,b.pid,b.ad_type_id,b.ad_type,b.click_num,datediff(str_to_date(b.revenue_date,'%%Y%%m%%d'),str_to_date(b.date,'%%Y%%m%%d')) as day_diff from (select date,max(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type from ROI_News_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and revenue_date<" + self.today + " and click_num>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) a,ROI_News_Predict_GroupBy_c b where b.pdtid=%s and a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.revenue_date=b.revenue_date and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type;"
            elif self.colname == "imp_revenue":
            
                self.insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_revenue_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE imp_revenue_predict=%s;"
                self.selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,a.ad_type_id,a.ad_type,datediff(str_to_date(a.revenue_date,'%%Y%%m%%d'),str_to_date(b.revenue_date,'%%Y%%m%%d')) as day_diff,b.revenue_date as purchase_day_min,log(b.imp_revenue/a.imp_revenue) as log_purchase_rate,b.imp_revenue from (select * from ROI_News_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and revenue_date<" + self.today + " and imp_revenue>0) a,(select date,min(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_revenue from ROI_News_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and revenue_date<" + self.today + " and imp_revenue>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type) d where day_diff>0 and log_purchase_rate is not null;"
                self.predictSelectSql = "select b.date,b.revenue_date,b.pdtid,b.country_iso,b.channel_category,b.pid,b.ad_type_id,b.ad_type,b.imp_revenue,datediff(str_to_date(b.revenue_date,'%%Y%%m%%d'),str_to_date(b.date,'%%Y%%m%%d')) as day_diff from (select date,max(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type from ROI_News_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and revenue_date<" + self.today + " and imp_revenue>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) a,ROI_News_Predict_GroupBy_c b where b.pdtid=%s and a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.revenue_date=b.revenue_date and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type;"
            elif self.colname == "click_revenue":
                self.insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,click_revenue_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE click_revenue_predict=%s;"
                self.selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,a.ad_type_id,a.ad_type,datediff(str_to_date(a.revenue_date,'%%Y%%m%%d'),str_to_date(b.revenue_date,'%%Y%%m%%d')) as day_diff,b.revenue_date as purchase_day_min,log(b.click_revenue/a.click_revenue) as log_purchase_rate,b.click_revenue from (select * from ROI_News_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and revenue_date<" + self.today + " and click_revenue>0) a,(select date,min(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,click_revenue from ROI_News_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and revenue_date<" + self.today + " and click_revenue>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type) d where day_diff>0 and log_purchase_rate is not null;"
                self.predictSelectSql = "select b.date,b.revenue_date,b.pdtid,b.country_iso,b.channel_category,b.pid,b.ad_type_id,b.ad_type,b.click_revenue,datediff(str_to_date(b.revenue_date,'%%Y%%m%%d'),str_to_date(b.date,'%%Y%%m%%d')) as day_diff from (select date,max(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type from ROI_News_Predict_GroupBy_c where pdtid=%s and date>=" + self.daysAgo60 + " and revenue_date<" + self.today + " and click_revenue>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) a,ROI_News_Predict_GroupBy_c b where b.pdtid=%s and a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.revenue_date=b.revenue_date and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type;"

        else:
            print "Please write right table name!"
            exit()

    def initData(self):
        if self.colname == "retain" or self.table == "ROI_Purchase_Predict_GroupBy_c":
            db = baseModel().conn()
            cursor = db.cursor()
            cursor.execute(self.pdtidSql)
            pdtidList = cursor.fetchall()
            print pdtidList
            for pdtid in pdtidList:
                print pdtid[0]
                sql = self.exeSql % (pdtid[0], pdtid[0])
                cursor.execute(sql)
                cursor.execute("commit;")

        elif self.colname == "imp_num":
            db = baseModel().conn()
            cursor = db.cursor()
            cursor.execute(self.pdtidSql)
            print self.pdtidSql
            pdtidList = cursor.fetchall()
            print pdtidList
            for pdtid in pdtidList:
                print pdtid[0]
                sql = self.exeSql % (pdtid[0], pdtid[0], pdtid[0], pdtid[0], pdtid[0], pdtid[0])
                cursor.execute(sql)
                cursor.execute("commit;")

        print "init data ok"

        baseModel().close()
        
    def predictResult(self, predictDic, key, keys, pDays, y_predict, cursor):
        for predictDate in predictDic[key]:
            print ("######### currentTime is %s") % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
            print predictDate
            if getPlusDay(predictDate[0],int(predictDate[2])) < self.daysAgo3:
                continue
            beginIndex = int(predictDate[2])
            if beginIndex >= pDays:
                continue
            if beginIndex == 0 :
                midResult = float(predictDate[1])
                #print midResult
            else:
                midResult = float(predictDate[1]) * y_predict[beginIndex-1]
                #print y_predict[0]
                #print y_predict[beginIndex-1]
                #print midResult
            for i in range(beginIndex, pDays):
                dateStr = getPlusDay(predictDate[0], i+1)
                result = midResult/y_predict[i]
                #print result
    
                if self.table == "ROI_Retained_Predict_GroupBy_c":
                    resultValues = [predictDate[0], dateStr, keys[0], keys[1], keys[2], keys[3], result, result]
                elif self.table == "ROI_Purchase_Predict_GroupBy_c":
                    resultValues = [predictDate[0], dateStr, keys[0], keys[1], keys[2], keys[3], result, result]
                elif self.table == "ROI_install_Predict_GroupBy_c":
                    resultValues = [dateStr, keys[0], keys[1], keys[2], keys[3], result, result]
                elif self.table == "ROI_News_Predict_GroupBy_c":
                    resultValues = [predictDate[0], dateStr, keys[0], keys[1], keys[2], keys[3], keys[4], keys[5], result, result]
                print resultValues
                
                cursor.execute(self.insertSql, resultValues)
            cursor.execute("commit;")
            #根据预测数据的有效长度，利用时间序列继续往后预测。
            predictLen = pDays - beginIndex
            predictRemain = self.predictDays - predictLen
            #根据key列表构造where条件
            whereSql = ''
            for j in range(len(self.keyName)):
                whereSql += self.keyName[j] + "='" + keys[j] + "' and "
            #添加日期条件
            whereSql += "date='" + predictDate[0] + "'"
            #获取已知数据的结束日期和开始日期
            endDay = getPlusDay(predictDate[0], pDays)
            beginDay = getPlusDay(endDay, -30)

            #if predictDate[0] == '20170207':
            #ROI_ARIMA = ARIMAModel(self.table, whereSql, predictRemain, beginDay, endDay, self.colname)
            #ROI_ARIMA.predict()

    def buildComplexDataset(self, x):
        temp = np.log(x)
        X = np.hstack((x,temp))
        temp = x * x
        X = np.hstack((X,temp))
        temp = temp * x
        X = np.hstack((X,temp))
        temp = 1.0/x
        X = np.hstack((X,temp))
        temp = temp/x
        X = np.hstack((X,temp))
        return X

    def skipThreshold(self, m, n):
        #print m, n, n*(n-1)/2.0*self.threshold
        if m<10:
            return True
        return True if m < n*(n-1)/2.0*self.threshold else False

    def getData(self, sql):
        result = ()
        db = baseModel().conn()
        cursor = db.cursor()
        cursor.execute(self.pdtidSql)
        pdtidList = cursor.fetchall()
        print pdtidList
        for pdtid in pdtidList:
            print pdtid[0]
            sqlresult = sql % (pdtid[0], pdtid[0])
            cursor.execute(sqlresult)
            data = cursor.fetchall()
            result = result + data

        baseModel().close()
        return result


    def predict(self):
        #获取预测的原始数据
        data = self.getData(self.selectSql)
        print "get select data ok"
        predictData = self.getData(self.predictSelectSql)
        print "get predict data ok"
        dataDic = getDic(data, self.keyList, self.valueList)
        
        predictDic = getDic(predictData, self.predictKeyList, self.predictValueList)

        lenDic = len(dataDic)
        keyIndex = 1
        currentTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        print ("######## Now is %s,All keys cnt is %d") % (currentTime, lenDic)
        for key in dataDic:
            #print predictDic[key]
            if self.skipThreshold(len(dataDic[key]), len(predictDic[key])):
                continue
            keys = key.split("#")
            #获取对应key的训练数据
            value = dataDic[key]
            if len(value) < 20:
            	continue
            x=[]
            y=[]
            for i in xrange(len(value)):
                x.append(int(value[i][0]))
                y.append(float(value[i][1]))
        
            x=np.array(x)
            x=x.reshape(x.shape[0],1)
            y=np.array(y)
        
            #构建较为复杂的训练数据
            X = self.buildComplexDataset(x)
            scaler = preprocessing.StandardScaler().fit(X)
            X = scaler.transform(X)
            #print X

            #构建模型
            alpha = 0.1
            l1_ratio = 0.8
            enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            model = enet.fit(X, y)
            #print ("######### currentTime is %s") % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
            #print model.coef_
        
            y_pred_lasso = model.predict(X) 
            r2_score_lasso = r2_score(y, y_pred_lasso)
            #print("r^2 on test data : %f" % r2_score_lasso)
        
            #预测过程
            #找到可预测的最大长度
            valueKey = [int(i[0]) for i in value]
            valueKey.sort()
            maxLen = valueKey[-1]
        
            if maxLen>self.predictDays-1:
                pDays = self.predictDays
            else:
                pDays =  maxLen
        
            X_predict = np.array(range(1, pDays + 1))
            X_predict = X_predict.reshape(X_predict.shape[0],1)
            X_predict = self.buildComplexDataset(X_predict)
            X_predict = scaler.transform(X_predict)
            y_predict = model.predict(X_predict) 
            y_predict = np.exp(y_predict)
            #print 'y_predict'
            #print y_predict
            predictDateList = predictDic[key]
            qList = [keyIndex, predictDateList, key, keys, pDays, y_predict]
            #print qList
            inputQ(self.q, qList)
            keyIndex = keyIndex + 1
            #self.predictResult(predictDic, key, keys, pDays, y_predict, cursor)

        # 关闭数据库连接
        baseModel().close()

class workProcess(Process):
    """
    docstring for workProcess
    #干活进程
    """
    def __init__(self, queue, lock, tableName, predictDays, colname):
        super(workProcess, self).__init__()
        self.queue = queue
        self.lock  = lock
        self.reg = RegModel(tableName, predictDays, colname, self.queue)
        self.db = baseModel().conn()
        self.cursor = self.db.cursor()
        
    def run(self):
        while True:
            rs = outputQ(self.queue, self.lock)
            if not rs:
                print "Process exit"
                sys.exit(0)
            else:
                keyIndex, predictDateList, key, keys, pDays, y_predict = rs
                currentTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                print ("######## Now is %s,The index %d key is %s") % (currentTime, keyIndex, key)
                for predictDate in predictDateList:
                    print ("######### currentTime is %s") % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
                    print predictDate
                    if getPlusDay(predictDate[0],int(predictDate[2])) < self.reg.daysAgo3:
                        continue
                    beginIndex = int(predictDate[2])
                    if beginIndex >= pDays:
                        continue
                    if beginIndex == 0 :
                        midResult = float(predictDate[1])
                        #print midResult
                    else:
                        midResult = float(predictDate[1]) * y_predict[beginIndex-1]
                    for i in range(beginIndex, pDays):
                        dateStr = getPlusDay(predictDate[0], i+1)
                        result = midResult/y_predict[i]
                        
            
                        if self.reg.table == "ROI_Retained_Predict_GroupBy_c":
                            resultValues = [predictDate[0], dateStr, keys[0], keys[1], keys[2], keys[3], result, result]
                        elif self.reg.table == "ROI_Purchase_Predict_GroupBy_c":
                            resultValues = [predictDate[0], dateStr, keys[0], keys[1], keys[2], keys[3], result, result]
                        elif self.reg.table == "ROI_install_Predict_GroupBy_c":
                            resultValues = [dateStr, keys[0], keys[1], keys[2], keys[3], result, result]
                        elif self.reg.table == "ROI_News_Predict_GroupBy_c":
                            resultValues = [predictDate[0], dateStr, keys[0], keys[1], keys[2], keys[3], keys[4], keys[5], result, result]
                        #print resultValues
                        
                        self.cursor.execute(self.reg.insertSql, resultValues)
                    self.cursor.execute("commit;")
                    #根据预测数据的有效长度，利用时间序列继续往后预测。
                    predictLen = pDays - beginIndex
                    predictRemain = self.reg.predictDays - predictLen
                    #根据key列表构造where条件
                    whereSql = ''
                    for j in range(len(self.reg.keyName)):
                        whereSql += self.reg.keyName[j] + "='" + keys[j] + "' and "
                    #添加日期条件
                    whereSql += "date='" + predictDate[0] + "'"
                    #获取已知数据的结束日期和开始日期
                    endDay = getPlusDay(predictDate[0], pDays)
                    beginDay = getPlusDay(endDay, -30)
                    #if predictDate[0] == '20170207':
                    #ROI_ARIMA = ARIMAModel(self.table, whereSql, predictRemain, beginDay, endDay, self.colname)
                    #ROI_ARIMA.predict()

def entrance(tableName, predictDays, colname):
    q = Queue()
    l = Lock()
    regModel = RegModel(tableName, predictDays, colname, q)
    regModel.initData()
    regModel.predict()

    joinList = []
    for i in range(PROCESS_NUM):
        w = workProcess(q, l, tableName, predictDays, colname)
        w.start()
        joinList.append(w)

    for p in joinList:
        p.join()

