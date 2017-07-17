# -*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pandas as pd
import numpy as np 
from copy import deepcopy
import MySQLdb
import datetime
import time
from math import isnan
import math
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

from sklearn import preprocessing
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
import ROI_model

if len(sys.argv) < 2:
   print "Please write table name!"
   exit() 

tableName = sys.argv[1]
print tableName
today = datetime.date.today().strftime("%Y%m%d")
print "begin"
#init

keyList = [] #从mysql中获取的数据中，这些列作为key，每一个表的初始化方式不一样
valueList = [] #从mysql中获取的数据中，这些列作为value
#这些列记录了待预测的信息
predictKeyList = []
predictValueList = []

#预测天数
predictDays = 90

if tableName == "ROI_Retained_Predict_GroupBy_c":
    exeSql = "INSERT INTO ROI_Retained_Predict_GroupBy_c (`date`, retain_day, pdtid, country_iso, channel_category, pid, retain) select date,retain_day,pdtid,country_iso,channel_category,pid,retain from (select date,retain_day,pdtid,country_iso,channel_category,pid,sum(retain) as retain from ROI_Retained group by date,retain_day,pdtid,country_iso,channel_category,pid) t on DUPLICATE KEY UPDATE retain=t.retain;"
    keyList = [1,2,3,4]
    valueList = [5,7]

    predictKeyList = [2,3,4,5]
    predictValueList = [0,6,7]

    keyName = ['pdtid', 'country_iso', 'channel_category', 'pid']

    insertSql = "INSERT INTO ROI_Retained_Predict_GroupBy_c (`date`, retain_day, pdtid, country_iso,channel_category, pid, retain_predict) VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE retain_predict=%s;"
    #获取经过处理的mysql数据
    #selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,datediff(str_to_date(a.retain_day,'%Y%m%d'),str_to_date(b.retain_day,'%Y%m%d')) as day_diff,b.retain_day as retain_day_min,log(b.retain/a.retain) as log_retain_rate,b.retain from (select * from ROI_Retained_Predict_GroupBy_c where retain>0) a,(select date,min(retain_day) as retain_day,pdtid,country_iso,channel_category,pid,retain as retain from ROI_Retained_Predict_GroupBy_c where retain>0 group by date,pdtid,country_iso,channel_category,pid) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid) d where day_diff>0 and log_retain_rate is not null;"
    #调试sql
    selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,datediff(str_to_date(a.retain_day,'%Y%m%d'),str_to_date(b.retain_day,'%Y%m%d')) as day_diff,b.retain_day as retain_day_min,log(b.retain/a.retain) as log_retain_rate,b.retain from (select * from ROI_Retained_Predict_GroupBy_c where country_iso='US' and pdtid='600001' and channel_category='affiliate' and pid='applovin_int' and retain>0) a,(select date,min(retain_day) as retain_day,pdtid,country_iso,channel_category,pid,retain as retain from ROI_Retained_Predict_GroupBy_c where retain>0 group by date,pdtid,country_iso,channel_category,pid) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid) d where day_diff>0 and log_retain_rate is not null;"

    predictSelectSql = "select b.date,b.retain_day,b.pdtid,b.country_iso,b.channel_category,b.pid,b.retain,datediff(str_to_date(b.retain_day,'%Y%m%d'),str_to_date(b.date,'%Y%m%d')) as day_diff from (select date,max(retain_day) as retain_day,pdtid,country_iso,channel_category,pid from ROI_Retained_Predict_GroupBy_c where retain>0 group by date,pdtid,country_iso,channel_category,pid) a,ROI_Retained_Predict_GroupBy_c b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.retain_day=b.retain_day;"

elif tableName == "ROI_install_Predict_GroupBy_c":
    keyList = [1,2,3,4]
    valueList = [5,7]

    predictKeyList = [1,2,3,4]
    predictValueList = [0,5,6]

    insertSql = "INSERT INTO ROI_install_Predict_GroupBy_c (`date`, pdtid, country_iso, channel_category, pid, install_num_predict) VALUES (%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE install_num_predict = %s"
    selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,datediff(str_to_date(a.date,'%Y%m%d'),str_to_date(b.date,'%Y%m%d')) as day_diff,b.date as install_day_min,log(b.install_num/a.install_num) as log_install_rate,b.install_num from (select * from ROI_install_Predict_GroupBy_c where install_num>0) a,(select min(date) as date,pdtid,country_iso,channel_category,pid,install_num from ROI_install_Predict_GroupBy_c where install_num>0 group by pdtid,country_iso,channel_category,pid) b where a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid) d where day_diff>0 and log_install_rate is not null;"
    # 调试sql
    #selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,datediff(str_to_date(a.date,'%Y%m%d'),str_to_date(b.date,'%Y%m%d')) as day_diff,b.date as install_day_min,log(b.install_num/a.install_num) as log_install_rate,b.install_num from (select * from ROI_install_Predict_GroupBy_c where pdtid=600018 and country_iso='US' and channel_category='FB' and pid='Facebook Ads' and install_num>0) a,(select min(date) as date,pdtid,country_iso,channel_category,pid,install_num from ROI_install_Predict_GroupBy_c where install_num>0 group by pdtid,country_iso,channel_category,pid) b where a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid) d where day_diff>0 and log_install_rate is not null;"

    predictSelectSql = "select * from (select b.date,b.pdtid,b.country_iso,b.channel_category,b.pid,a.install_num,datediff(str_to_date(a.date,'%Y%m%d'),str_to_date(b.date,'%Y%m%d')) as day_diff from (select max(date) as date,pdtid,country_iso,channel_category,pid,install_num from ROI_install_Predict_GroupBy_c where install_num>0 group by pdtid,country_iso,channel_category,pid) a,(select min(date) as date,pdtid,country_iso,channel_category,pid from ROI_install_Predict_GroupBy_c where install_num>0 group by pdtid,country_iso,channel_category,pid) b where a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid) c where day_diff>0;"

elif tableName == "ROI_Purchase_Predict_GroupBy_c":
    exeSql = "INSERT INTO ROI_Purchase_Predict_GroupBy_c (`date`, revenue_day, pdtid, country_iso, channel_category, pid, purchase) select date,revenue_day,pdtid,country_iso,channel_category,pid,purchase from (select date,revenue_day,pdtid,country_iso,channel_category,pid,sum(purchase) as purchase from ROI_Purchase group by date,revenue_day,pdtid,country_iso,channel_category,pid) t on DUPLICATE KEY UPDATE purchase=t.purchase;"
    keyList = [1,2,3,4]
    valueList = [5,7]

    predictKeyList = [2,3,4,5]
    predictValueList = [0,6,7]

    insertSql = "INSERT INTO ROI_Purchase_Predict_GroupBy_c (`date`,revenue_day,pdtid,country_iso,channel_category,pid,purchase_predict) VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE purchase_predict=%s;"
    selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,datediff(str_to_date(a.revenue_day,'%Y%m%d'),str_to_date(b.revenue_day,'%Y%m%d')) as day_diff,b.revenue_day as purchase_day_min,log(b.purchase/a.purchase) as log_purchase_rate,b.purchase from (select * from ROI_Purchase_Predict_GroupBy_c where purchase>0) a,(select date,min(revenue_day) as revenue_day,pdtid,country_iso,channel_category,pid,purchase from ROI_Purchase_Predict_GroupBy_c where purchase>0 group by date,pdtid,country_iso,channel_category,pid) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid) d where day_diff>0 and log_purchase_rate is not null;"
    predictSelectSql = "select b.date,b.revenue_day,b.pdtid,b.country_iso,b.channel_category,b.pid,b.purchase,datediff(str_to_date(b.revenue_day,'%Y%m%d'),str_to_date(b.date,'%Y%m%d')) as day_diff from (select date,max(revenue_day) as revenue_day,pdtid,country_iso,channel_category,pid from ROI_Purchase_Predict_GroupBy_c where purchase>0 group by date,pdtid,country_iso,channel_category,pid) a,ROI_Purchase_Predict_GroupBy_c b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.revenue_day=b.revenue_day;"
else:
    print "Please write right table name!"
    exit()


# 打开数据库连接
db = MySQLdb.connect("test-dataverse-web.ccxsyvw76h8l.rds.cn-north-1.amazonaws.com.cn","datauser","ebOUeWSin3hAAKKD","dataverse" ) 
cursor = db.cursor()
cursor.execute(exeSql)
cursor.execute("commit;")
cursor.execute(selectSql)
data = cursor.fetchall()
cursor.execute(predictSelectSql)
predictData = cursor.fetchall()

def getDic(data, keyList, valueList):
    dataDic={} #是一个规划数据的字典，字典key有产品号和渠道号联合组成
    for i in range(len(data)):
        key=""
        for j in keyList:
            key = key + str(data[i][j]) + '#'
            #key=str(data[i][0]) + '#' + str(data[i][2]) + '#' + str(data[i][3]) + '#' + str(data[i][4]) + '#' + str(data[i][5])
        key = key.strip("#")
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

def predictResult(predictDic, key, keys, pDays):
    for predictDate in predictDic[key]:
        print 'predictDate'
        print predictDate
        print key
        print pDays
        beginIndex = int(predictDate[2])
        if beginIndex >= pDays:
            continue
        if beginIndex == 1:
            midResult = float(predictDate[1])
        else:
            midResult = float(predictDate[1]) * y_predict[beginIndex-2]
        for i in range(beginIndex, pDays):
            dateStr = getPlusDay(predictDate[0], i+1)
            result = midResult/y_predict[i-1]
            #print result

            if tableName == "ROI_Retained_Predict_GroupBy_c":
                resultValues = [predictDate[0], dateStr, keys[0], keys[1], keys[2], keys[3], result, result]
            elif tableName == "ROI_Purchase_Predict_GroupBy_c":
                resultValues = [predictDate[0], dateStr, keys[0], keys[1], keys[2], keys[3], result, result]
            elif tableName == "ROI_install_Predict_GroupBy_c":
                resultValues = [dateStr, keys[0], keys[1], keys[2], keys[3], result, result]
            cursor.execute(insertSql, resultValues)
        cursor.execute("commit;")
        #根据预测数据的有效长度，利用时间序列继续往后预测。
        predictLen = pDays - beginIndex
        predictRemain = predictDays - predictLen
        #根据key列表构造where条件
        whereSql = ''
        for j in range(len(keyName)):
            whereSql += keyName[j] + "='" + keys[j] + "' and "
        #添加日期条件
        whereSql += "date='" + predictDate[0] + "'"
        #获取已知数据的结束日期和开始日期
        endDay = getPlusDay(predictDate[0], pDays)
        beginDay = getPlusDay(endDay, -30)

        print whereSql
        print beginDay
        print endDay
        ROI_ARIMA = ROI_model.ARIMAModel(tableName, whereSql, predictRemain, beginDay, endDay)
        ROI_ARIMA.predict()



def buildComplexDataset(x):
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


dataDic = getDic(data, keyList, valueList)
#print dataDic
predictDic = getDic(predictData, predictKeyList, predictValueList)

for key in dataDic:
    print key
    length = len(dataDic[key])
    keys = key.split("#")
    #获取训练数据
    value = dataDic[key]
    x=[]
    y=[]
    for i in xrange(len(value)):
        x.append(int(value[i][0]))
        y.append(float(value[i][1]))

    x=np.array(x)
    x=x.reshape(x.shape[0],1)
    y=np.array(y)

    #构建较为复杂的训练数据
    X = buildComplexDataset(x)
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    #print X

    #构建模型
    alpha = 0.1
    l1_ratio = 0.8
    enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model = enet.fit(X, y)
    print model.coef_

    y_pred_lasso = model.predict(X) 
    r2_score_lasso = r2_score(y, y_pred_lasso)
    print("r^2 on test data : %f" % r2_score_lasso)

    #reg = Ridge(alpha=0.5)
    #model2 = reg.fit(X, y)
    #print model2.coef_

    #y_pred_lasso = model2.predict(X) 
    #r2_score_lasso = r2_score(y, y_pred_lasso)
    #print("r^2 on test data : %f" % r2_score_lasso)
    #print("\n")

    #预测过程
    #找到可预测的最大长度
    valueKey = [int(i[0]) for i in value]
    valueKey.sort()
    maxLen = valueKey[-1]

    if maxLen>predictDays-1:
        pDays = predictDays
    else:
        pDays =  maxLen + 3

    X_predict = np.array(range(1, pDays + 1))
    X_predict = X_predict.reshape(X_predict.shape[0],1)
    X_predict = buildComplexDataset(X_predict)
    X_predict = scaler.transform(X_predict)
    y_predict = model.predict(X_predict) 
    y_predict = np.exp(y_predict)

    predictResult(predictDic, key, keys, pDays)
    

cursor.execute("commit;")
# 关闭数据库连接
db.close()

print("end.")



