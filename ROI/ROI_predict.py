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
from dateutil.relativedelta import relativedelta

if len(sys.argv) < 2:
   print "Please write table name!"
   exit() 

tableName = sys.argv[1]
print tableName
today = datetime.date.today().strftime("%Y%m%d")
print "begin"
#init
dic = {}
keyList = []
valueList = []
exeSql = ""
insertSql = ""
if tableName == "ROI_Retained_Predict_GroupBy_c":
    exeSql = "INSERT INTO ROI_Retained_Predict_GroupBy_c (`date`, retain_day, pdtid, country_iso, channel_category, pid, retain) select date,retain_day,pdtid,country_iso,channel_category,pid,retain from (select date,retain_day,pdtid,country_iso,channel_category,pid,sum(retain) as retain from ROI_Retained group by date,retain_day,pdtid,country_iso,channel_category,pid) t on DUPLICATE KEY UPDATE retain=t.retain;"
    keyList = [0,2,3,4,5]
    valueList = [1,6]
    insertSql = "INSERT INTO ROI_Retained_Predict_GroupBy_c (`date`, retain_day, pdtid, country_iso,channel_category, pid, retain_predict) VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE retain_predict=%s;"
    selectSql = "select * from ROI_Retained_Predict_GroupBy_c where retain>0 and retain_day<" + today + ";"
    #selectSql = "select * from ROI_Retained_Predict_GroupBy_c where retain>0 and date='20161212' and pdtid='600001' and  retain_day<" + today + ";"
elif tableName == "ROI_install_Predict_GroupBy_c":
    exeSql = "INSERT INTO ROI_install_Predict_GroupBy_c (`date`, pdtid, country_iso, channel_category, pid, install_num) select date,pdtid,country_iso,channel_category,pid,install_num from (select date,pdtid,country_iso,channel_category,pid,sum(install_num) as install_num from ROI_install group by date,pdtid,country_iso,channel_category,pid) t on DUPLICATE KEY UPDATE install_num=t.install_num;"
    keyList = [1,2,3,4]
    valueList = [0,5]
    insertSql = "INSERT INTO ROI_install_Predict_GroupBy_c (`date`, pdtid, country_iso, channel_category, pid, install_num_predict) VALUES (%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE install_num_predict = %s"
    selectSql = "select * from ROI_install_Predict_GroupBy_c where install_num>0 and date<" + today + ";"
elif tableName == "ROI_Purchase_Predict_GroupBy_c":
    exeSql = "INSERT INTO ROI_Purchase_Predict_GroupBy_c (`date`, revenue_day, pdtid, country_iso, channel_category, pid, purchase) select date,revenue_day,pdtid,country_iso,channel_category,pid,purchase from (select date,revenue_day,pdtid,country_iso,channel_category,pid,sum(purchase) as purchase from ROI_Purchase group by date,revenue_day,pdtid,country_iso,channel_category,pid) t on DUPLICATE KEY UPDATE purchase=t.purchase;"
    keyList = [0,2,3,4,5]
    valueList = [1,6]
    insertSql = "INSERT INTO ROI_Purchase_Predict_GroupBy_c (`date`,revenue_day,pdtid,country_iso,channel_category,pid,purchase_predict) VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE purchase_predict=%s;"
    selectSql = "select * from ROI_Purchase_Predict_GroupBy_c where purchase>0 and revenue_day<" + today + ";"
    #selectSql = "select * from ROI_Purchase_Predict_GroupBy_c where pdtid='600001' and country_iso='AU' and channel_category='affiliate' and pid='applovin_int' and date='20161118' and  purchase>0 and revenue_day<" + today + ";"
else:
    print "Please write right table name!"
    exit()

windowSize = 7
forecastnum = 90

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
    print(u'BIC最小的p值和q值为：%s、%s' %(p,q)) 
    return p, c, q



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
#print dic

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
    #print td
    indexs = []
    cnt = []
    if td == length:
        for j in range(td):
            indexs.append(value[j][0])
            cnt.append(value[j][1])
    #处理连续日期，中间有数据缺失的情况
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
    
    if length>6:
        resultValues = []
        datats = data[2:]
        #print datats
        newDate = datats.index[-1] + relativedelta(days=1)
        endData = datats.index[-1] + relativedelta(days=forecastnum)
        newDateStr = newDate.strftime("%Y-%m-%d")
        endDataStr = endData.strftime("%Y-%m-%d")
        #print datats #a=ARIMA(datats, (0,0,0)).fit() #print a.predict(start='2016-11-03',end='2016-11-03')
        result = buildARIMA(datats,newDateStr,endDataStr)
        #print result
        for i in range(forecastnum):
            dateStr = (newDate + relativedelta(days=i)).strftime("%Y-%m-%d").replace("-","")
            if tableName == "ROI_Retained_Predict_GroupBy_c":
                resultValues = [keys[0], dateStr, keys[1], keys[2], keys[3], keys[4], result[i], result[i]]
            elif tableName == "ROI_Purchase_Predict_GroupBy_c":
                resultValues = [keys[0], dateStr, keys[1], keys[2], keys[3], keys[4], result[i], result[i]]
            elif tableName == "ROI_install_Predict_GroupBy_c":
                resultValues = [dateStr, keys[0], keys[1], keys[2], keys[3], result[i], result[i]]
            #print resultValues
            #print insertSql
            cursor.execute(insertSql, resultValues)
        cursor.execute("commit;")

cursor.execute("commit;")
# 关闭数据库连接
db.close()

print("end.")



