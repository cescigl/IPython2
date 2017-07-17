# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import datetime
import MySQLdb
import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, median_absolute_error

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

db = baseModel().conn()
cursor = db.cursor()

def getData(sql):
    cursor.execute(sql)
    data = cursor.fetchall()
    return data

def main():
    #sql_new = "select date,retain_day,pdtid,retain,retain_predict from ROI_Retained_Predict_GroupBy_c_zhx where  date>=20170413 and country_iso='' and pid='' and channel_category='' and retain>0 and retain_predict>0;"
    #sql_his = "select date,retain_day,pdtid,cnt,cnt_predict from (select date,retain_day,pdtid,CAST(sum(retain) AS SIGNED) as cnt ,sum(retain_predict) as cnt_predict from ROI_Retained_Predict_GroupBy_c where date>=20170413 and pdtid in (400125,400126,400129,600001,600004,600007,600008,600018,600020,600022,600025,600027,600029,600030) group by date,retain_day,pdtid) t where cnt>0.0 and cnt_predict>0;"

    sql1 = "select t1.retain,t1.retain_predict,t2.cnt,t2.cnt_predict from (select date,retain_day,pdtid,retain,retain_predict from ROI_Retained_Predict_GroupBy_c_zhx where  date>=20170413 and country_iso='' and pid='' and channel_category='' and retain>0 and retain_predict>0 and retain_day<=20170418) t1,(select date,retain_day,pdtid,cnt,cnt_predict from (select date,retain_day,pdtid,sum(retain) as cnt ,sum(retain_predict) as cnt_predict from ROI_Retained_Predict_GroupBy_c where date>=20170413 and pdtid in (400125,400126,400129,600001,600004,600007,600008,600018,600020,600022,600025,600027,600029,600030) group by date,retain_day,pdtid) t where cnt>0.0 and cnt_predict>0) t2 where t1.date=t2.date and t1.retain_day=t2.retain_day and t1.pdtid=t2.pdtid;"

    sql2 = "select t1.retain,t1.retain_predict,t2.cnt,t2.cnt_predict from (select date,retain_day,pdtid,country_iso,retain,retain_predict from ROI_Retained_Predict_GroupBy_c_zhx where  date>=20170413 and country_iso='US' and pid='' and channel_category='' and retain>0 and retain_predict>0 and retain_day<=20170418) t1,(select date,retain_day,pdtid,country_iso,cnt,cnt_predict from (select date,retain_day,pdtid,country_iso,sum(retain) as cnt ,sum(retain_predict) as cnt_predict from ROI_Retained_Predict_GroupBy_c where date>=20170413 and country_iso='US' and pdtid in (400125,400126,400129,600001,600004,600007,600008,600018,600020,600022,600025,600027,600029,600030) group by date,retain_day,pdtid,country_iso) t where cnt>0.0 and cnt_predict>0) t2 where t1.date=t2.date and t1.retain_day=t2.retain_day and t1.pdtid=t2.pdtid and t1.country_iso=t2.country_iso;" 
    sql3 = "select t1.retain,t1.retain_predict,t2.cnt,t2.cnt_predict from (select date,retain_day,pdtid,pid,retain,retain_predict from ROI_Retained_Predict_GroupBy_c_zhx where date>=20170413 and country_iso='' and pid='Organic' and retain>0 and retain_predict>0 and retain_day<=20170418) t1,(select date,retain_day,pdtid,pid,cnt,cnt_predict from (select date,retain_day,pdtid,pid,sum(retain) as cnt ,sum(retain_predict) as cnt_predict from ROI_Retained_Predict_GroupBy_c where date>=20170413 and pid='Organic' and pdtid in (400125,400126,400129,600001,600004,600007,600008,600018,600020,600022,600025,600027,600029,600030) group by date,retain_day,pdtid,pid) t where cnt>0.0 and cnt_predict>0) t2 where t1.date=t2.date and t1.retain_day=t2.retain_day and t1.pdtid=t2.pdtid and t1.pid=t2.pid;"
    sql = sql1

    data = getData(sql)
    y_true_new = [float(i[0]) for i in data]
    y_pred_new = [float(i[1]) for i in data]
    
    y_true_new = np.array(y_true_new) 
    y_pred_new = np.array(y_pred_new)

    y_true_his = [float(i[2]) for i in data]
    y_pred_his = [float(i[3]) for i in data]
    r2_score_new = r2_score(y_true_new, y_pred_new)
    print r2_score_new
    r2_score_his = r2_score(y_true_his, y_pred_his)
    print r2_score_his

    y_true_his = np.array(y_true_his)
    y_pred_his = np.array(y_pred_his)

    mean_squared_error(y_true_new, y_pred_new)/len(y_true_new)
    mean_squared_error(y_true_his, y_pred_his)/len(y_true_new)
    mean_absolute_error(y_true_new, y_pred_new)/len(y_true_new)
    mean_absolute_error(y_true_his, y_pred_his)/len(y_true_new)
    median_absolute_error(y_true_new, y_pred_new)/len(y_true_new)
    median_absolute_error(y_true_new, y_pred_his)/len(y_true_new)

    #error curve
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    ax1.scatter(y_true_new, y_pred_new, c='#F2BE2C')
    ax1.set_title('Prediction Error for %s' % "new")
    ax1.plot([y_true_new.min(), y_true_new.max()], [y_true_new.min(), y_true_new.max()], 'k--', lw=4, c='#2B94E9')
    ax1.set_ylabel('Predicted')

    ax2.scatter(y_true_his, y_pred_his, c='#F2BE2C')
    ax2.set_title('Prediction Error for %s' % "history")
    ax2.plot([y_true_new.min(), y_true_new.max()], [y_true_new.min(), y_true_new.max()], 'k--', lw=4, c='#2B94E9')
    ax2.set_ylabel('Predicted')

    plt.xlabel('Measured')
    plt.show()

    #残差曲线
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    plt.title('Plotting residuals')
    ax1.scatter(y_true_new, (y_pred_new-y_true_new)/y_true_new,c='#2B94E9',s=20,alpha=0.5)
    #ax1.hlines(y=0, xmin=0, xmax=100)
    ax1.set_title("new")
    ax1.set_ylabel('Residuals')

    ax2.scatter(y_true_new, (y_pred_his-y_true_new)/y_true_new,c='#2B94E9',s=20,alpha=0.5)
    #ax2.hlines(y=0, xmin=0, xmax=100)
    ax2.set_title("history")
    ax2.set_ylabel('Residuals')

    plt.xlim([y_true_new.min(), y_true_new.max()])
    plt.ylim([-2,2])  
    plt.show()



if __name__ == '__main__':
    main()
