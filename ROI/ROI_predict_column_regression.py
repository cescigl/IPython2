# -*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from copy import deepcopy
import MySQLdb
import time
import datetime
from math import isnan
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

from sklearn import preprocessing
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

if len(sys.argv) < 2:
   print "Please write column name!"
   exit() 

columnName = sys.argv[1]
print columnName
print "begin"
today = datetime.date.today().strftime("%Y%m%d")

#init
dic = {}
keyList = [1,2,3,4,5,6]
valueList = [7,9]

predictKeyList = [2,3,4,5,6,7]
predictValueList = [0,8,9]


exeSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_num,click_num,imp_revenue,click_revenue) select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_num,click_num,imp_revenue,click_revenue from (select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,sum(imp_num) as imp_num,sum(click_num) as click_num,sum(imp_revenue) as imp_revenue,sum(click_revenue) as click_revenue from ROI_News group by date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) t on DUPLICATE KEY UPDATE imp_num=t.imp_num,click_num=t.click_num,imp_revenue=t.imp_revenue,click_revenue=t.click_revenue;"

if columnName == "imp_num":
    insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_num_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE imp_num_predict=%s;"
    selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,a.ad_type_id,a.ad_type,datediff(str_to_date(a.revenue_date,'%Y%m%d'),str_to_date(b.revenue_date,'%Y%m%d')) as day_diff,b.revenue_date as purchase_day_min,log(b.imp_num/a.imp_num) as log_purchase_rate,b.imp_num from (select * from ROI_News_Predict_GroupBy_c where imp_num>0) a,(select date,min(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_num from ROI_News_Predict_GroupBy_c where imp_num>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type) d where day_diff>0 and log_purchase_rate is not null;"
    predictSelectSql = "select b.date,b.revenue_date,b.pdtid,b.country_iso,b.channel_category,b.pid,b.ad_type_id,b.ad_type,b.imp_num,datediff(str_to_date(b.revenue_date,'%Y%m%d'),str_to_date(b.date,'%Y%m%d')) as day_diff from (select date,max(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type from ROI_News_Predict_GroupBy_c where imp_num>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) a,ROI_News_Predict_GroupBy_c b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.revenue_date=b.revenue_date and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type;"
elif columnName == "click_num":
    insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,click_num_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE click_num_predict=%s;"
    selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,a.ad_type_id,a.ad_type,datediff(str_to_date(a.revenue_date,'%Y%m%d'),str_to_date(b.revenue_date,'%Y%m%d')) as day_diff,b.revenue_date as purchase_day_min,log(b.click_num/a.click_num) as log_purchase_rate,b.click_num from (select * from ROI_News_Predict_GroupBy_c where click_num>0) a,(select date,min(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,click_num from ROI_News_Predict_GroupBy_c where click_num>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type) d where day_diff>0 and log_purchase_rate is not null;"
    predictSelectSql = "select b.date,b.revenue_date,b.pdtid,b.country_iso,b.channel_category,b.pid,b.ad_type_id,b.ad_type,b.click_num,datediff(str_to_date(b.revenue_date,'%Y%m%d'),str_to_date(b.date,'%Y%m%d')) as day_diff from (select date,max(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type from ROI_News_Predict_GroupBy_c where click_num>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) a,ROI_News_Predict_GroupBy_c b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.revenue_date=b.revenue_date and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type;"
elif columnName == "imp_revenue":
    insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_revenue_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE imp_revenue_predict=%s;"
    selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,a.ad_type_id,a.ad_type,datediff(str_to_date(a.revenue_date,'%Y%m%d'),str_to_date(b.revenue_date,'%Y%m%d')) as day_diff,b.revenue_date as purchase_day_min,log(b.imp_revenue/a.imp_revenue) as log_purchase_rate,b.imp_revenue from (select * from ROI_News_Predict_GroupBy_c where imp_revenue>0) a,(select date,min(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_revenue from ROI_News_Predict_GroupBy_c where imp_revenue>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type) d where day_diff>0 and log_purchase_rate is not null;"
    predictSelectSql = "select b.date,b.revenue_date,b.pdtid,b.country_iso,b.channel_category,b.pid,b.ad_type_id,b.ad_type,b.imp_revenue,datediff(str_to_date(b.revenue_date,'%Y%m%d'),str_to_date(b.date,'%Y%m%d')) as day_diff from (select date,max(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type from ROI_News_Predict_GroupBy_c where imp_revenue>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) a,ROI_News_Predict_GroupBy_c b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.revenue_date=b.revenue_date and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type;"
elif columnName == "click_revenue":
    insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,click_revenue_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE click_revenue_predict=%s;"
    selectSql = "select * from (select a.date,a.pdtid,a.country_iso,a.channel_category,a.pid,a.ad_type_id,a.ad_type,datediff(str_to_date(a.revenue_date,'%Y%m%d'),str_to_date(b.revenue_date,'%Y%m%d')) as day_diff,b.revenue_date as purchase_day_min,log(b.click_revenue/a.click_revenue) as log_purchase_rate,b.click_revenue from (select * from ROI_News_Predict_GroupBy_c where click_revenue>0) a,(select date,min(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,click_revenue from ROI_News_Predict_GroupBy_c where click_revenue>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type) d where day_diff>0 and log_purchase_rate is not null;"
    predictSelectSql = "select b.date,b.revenue_date,b.pdtid,b.country_iso,b.channel_category,b.pid,b.ad_type_id,b.ad_type,b.click_revenue,datediff(str_to_date(b.revenue_date,'%Y%m%d'),str_to_date(b.date,'%Y%m%d')) as day_diff from (select date,max(revenue_date) as revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type from ROI_News_Predict_GroupBy_c where click_revenue>0 group by date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) a,ROI_News_Predict_GroupBy_c b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.revenue_date=b.revenue_date and a.ad_type_id=b.ad_type_id and a.ad_type=b.ad_type;"
else:
    print "Please write right column name!"
    exit()

#预测天数
predictDays = 90

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

def predictResult(predictDic, keys, predictDays):
    for predictDate in predictDic[key]:
        beginIndex = int(predictDate[2])
        if beginIndex >= predictDays:
            continueq
        if beginIndex == 1:
            midResult = float(predictDate[1])
        else:
            midResult = float(predictDate[1]) * y_predict[beginIndex-2]
        for i in range(beginIndex, predictDays):
            dateStr = getPlusDay(predictDate[0], i+1)
            result = midResult/y_predict[i-1]
            print result
            resultValues = [predictDate[0], dateStr, keys[0], keys[1], keys[2], keys[3], keys[4], keys[5], result, result]

            cursor.execute(insertSql, resultValues)
    cursor.execute("commit;")

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

    predictResult(predictDic, keys, pDays)
    

cursor.execute("commit;")
# 关闭数据库连接
db.close()

print("end.")