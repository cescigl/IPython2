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

if len(sys.argv) < 2:
   print "Please write column name!"
   exit() 

columnName = sys.argv[1]
print columnName
print "begin"
today = datetime.date.today().strftime("%Y%m%d")

#init
dic = {}
keyList = [0,2,3,4,5,6,7]
valueList = [1,8]
exeSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_num,click_num,imp_revenue,click_revenue) select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_num,click_num,imp_revenue,click_revenue from (select date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,sum(imp_num) as imp_num,sum(click_num) as click_num,sum(imp_revenue) as imp_revenue,sum(click_revenue) as click_revenue from ROI_News group by date,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type) t on DUPLICATE KEY UPDATE imp_num=t.imp_num,click_num=t.click_num,imp_revenue=t.imp_revenue,click_revenue=t.click_revenue;"
insertSql = ""
if columnName == "imp_num":
    insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_num_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE imp_num_predict=%s;"
    #selectSql = "select `date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_num from ROI_News_Predict_GroupBy_c where pdtid='600018' and ad_type_id='fb' and date='20161126' and country_iso='US' and channel_category='affiliate' and revenue_date<'20161204'"
    selectSql = "select `date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_num from ROI_News_Predict_GroupBy_c where imp_num>0 and revenue_date<" + today + ";"
elif columnName == "click_num":
    insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,click_num_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE click_num_predict=%s;"
    selectSql = "select `date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,click_num from ROI_News_Predict_GroupBy_c where click_num>0 and revenue_date<" + today + ";"
elif columnName == "imp_revenue":
    insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_revenue_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE imp_revenue_predict=%s;"
    selectSql = "select `date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_revenue from ROI_News_Predict_GroupBy_c where imp_revenue>0 and revenue_date<" + today + ";"
    #selectSql = "select `date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,imp_revenue from ROI_News_Predict_GroupBy_c where pdtid='600001' and ad_type='interstitial' and ad_type_id='fb' and date='20161215' and imp_revenue>0 and revenue_date<" + today + ";"
elif columnName == "click_revenue":
    insertSql = "INSERT INTO ROI_News_Predict_GroupBy_c (`date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,click_revenue_predict) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE click_revenue_predict=%s;"
    selectSql = "select `date`,revenue_date,pdtid,country_iso,channel_category,pid,ad_type_id,ad_type,click_revenue from ROI_News_Predict_GroupBy_c where click_revenue>0 and  revenue_date<" + today + ";"
else:
    print "Please write right column name!"
    exit()

windowSize = 7
forecastnum = 90
today = datetime.date.today().strftime("%Y%m%d")

# 打开数据库连接
db = MySQLdb.connect("test-dataverse-web.ccxsyvw76h8l.rds.cn-north-1.amazonaws.com.cn","datauser","ebOUeWSin3hAAKKD","dataverse" ) 
cursor = db.cursor()
cursor.execute(exeSql)
cursor.execute("commit;")
cursor.execute(selectSql)
data = cursor.fetchall()

from statsmodels.tsa.arima_model import ARIMA

#定阶
def definePQ(data):
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
                r = ARIMA(data, (p,c,q)).fit().bic
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
    print(u'BIC最小的p值和q值为：%s、%s' %(p,q)) 
    return p, c, q


# In[159]:

#建立ARIMA(p, c, q)模型
def buildARIMA(data, start, end):
    p,c,q = definePQ(data)
    model = ARIMA(data, (p,c,q)).fit()
    
    if c == 0:
        result = model.predict(start=start, end=end)
    else:
        result = model.predict(start=start, end=end, typ='levels')
    result = np.exp(result)
    return result



def getDic(data, keyList, valueList):
    dic={}
    for i in range(len(data)):
        key=""
        for j in keyList:
            key = key + str(data[i][j]) + '#'
            #key=str(data[i][0]) + '#' + str(data[i][2]) + '#' + str(data[i][3]) + '#' + str(data[i][4]) + '#' + str(data[i][5])
        key = key.strip("#")
        if dic.has_key(key):
            dic[key].extend([[data[i][valueList[0]],data[i][valueList[1]]]])
        else:
            dic[key]=[[data[i][valueList[0]],data[i][valueList[1]]]]
    return dic


dic = getDic(data, keyList, valueList)


def formatTime(strs):
    strs = str(strs)
    return strs[0:4] + '-' + strs[4:6] + '-' + strs[6:8]

def dataDiff(last, begin):
    date1=datetime.datetime.strptime(str(begin),"%Y%m%d")
    date2=datetime.datetime.strptime(str(last),"%Y%m%d")
    return (date2 - date1).days + 1

def getPlusDay(day, plus):
    date = datetime.datetime.strptime(str(day),"%Y%m%d") + datetime.timedelta(days=plus)
    string = date.strftime("%Y%m%d")
    return string


def getIndexCnt(value, td, length):
    indexs = []
    cnt = []
    if td == length:
        for j in range(td):
            indexs.append(value[j][0])
            cnt.append(value[j][1])
    else:
        j = 0
        begintime = str(value[0][0])
        for i in range(td):
            nowtime = getPlusDay(begintime ,i)
            if nowtime == str(value[j][0]):
                indexs.append(value[j][0])
                cnt.append(value[j][1])
                j+=1
            else:
                #如果数据连续缺失超过一天，则不预测。
                df=dataDiff(value[j][0],value[j-1][0])-1
                if df > 2:
                    return [],[]
                indexs.append(long(nowtime))
                cnt.append((value[j][1]+value[j-1][1])/2)

    return indexs, cnt

colname = ['cnt']

lenDic = len(dic)
keyIndex = 1
for key in dic:
    currentTime = time.strftime('%Y-%m-%d %X', time.localtime())
    print ("######## Now is %s,The index %d key is %s,All keys cnt is %d") % (currentTime, keyIndex, key, lenDic)
    keyIndex = keyIndex + 1

    length = len(dic[key])
    keys = key.split("#")
    value = dic[key]
    value.sort()
    #计算开始时间和结束时间的相隔天数
    last = value[-1][0]
    begin = value[0][0]
    td = dataDiff(last, begin)
    
    #获取索引和计数值
    indexs, cnt = getIndexCnt(value, td, length)
    if len(indexs) == 0:
        continue
    
    ts = pd.DataFrame(cnt, index=indexs, columns=colname)
    ts['day'] = ts.index
    ts.index = pd.to_datetime(ts['day'].apply(formatTime))
    data = np.log(ts['cnt'])

    if length>5:
        resultValues = []
        datats = data[1:]
        #print datats
        newDate = datats.index[-1] + relativedelta(days=1)
        endData = datats.index[-1] + relativedelta(days=forecastnum)
        newDateStr = newDate.strftime("%Y-%m-%d")
        endDataStr = endData.strftime("%Y-%m-%d")
        #print datats #a=ARIMA(datats, (0,0,0)).fit() #print a.predict(start='2016-11-03',end='2016-11-03')
        result = buildARIMA(datats,newDateStr,endDataStr)
        for i in range(forecastnum):
            dateStr = (newDate + relativedelta(days=i)).strftime("%Y-%m-%d").replace("-","")
            resultValues = [keys[0], dateStr, keys[1], keys[2], keys[3], keys[4], keys[5], keys[6], result[i], result[i]]
            cursor.execute(insertSql, resultValues)
        cursor.execute("commit;")

cursor.execute("commit;")
# 关闭数据库连接
db.close()

print("end.")
