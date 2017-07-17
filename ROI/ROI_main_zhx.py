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
    sql_new = "select date,retain_day,pdtid,retain,retain_predict from ROI_Retained_Predict_GroupBy_c_zhx where  date>=20170413 and country_iso='' and pid='' and channel_category='' and retain>0 and retain_predict>0;"
    sql_his = "select date,retain_day,pdtid,cnt,cnt_predict from (select date,retain_day,pdtid,CAST(sum(retain) AS SIGNED) as cnt ,sum(retain_predict) as cnt_predict from ROI_Retained_Predict_GroupBy_c where date>=20170413 and pdtid in (400125,400126,400129,600001,600004,600007,600008,600018,600020,600022,600025,600027,600029,600030) group by date,retain_day,pdtid) t where cnt>0.0 and cnt_predict>0;"

    data = getData(sql_new)
    y_true1 = [float(i[3]) for i in data]
    y_pred1 = [float(i[4]) for i in data]
    r2_score1 = r2_score(y_true1, y_pred1)
    y_true1 = np.array(y_true1) 
    y_pred1 = np.array(y_pred1)

    data = getData(sql_his)
    y_true1_cmp = [float(i[3]) for i in data]
    y_pred1_cmp = [float(i[4]) for i in data]
    r2_score1_cmp = r2_score(y_true1_cmp, y_pred1_cmp) 
    y_true1_cmp = np.array(y_true1_cmp)
    y_pred1_cmp = np.array(y_pred1_cmp)

    #error curve
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    ax1.scatter(y_true1, y_pred1, c='#F2BE2C')
    ax1.set_title('Prediction Error for %s' % "sql_new")
    ax1.plot([y_true1.min(), y_true1.max()], [y_true1.min(), y_true1.max()], 'k--', lw=4, c='#2B94E9')
    ax1.set_ylabel('Predicted')

    ax2.scatter(y_true1_cmp, y_pred1_cmp, c='#F2BE2C')
    ax2.set_title('Prediction Error for %s' % "sql_his")
    ax2.plot([y_true1.min(), y_true1.max()], [y_true1.min(), y_true1.max()], 'k--', lw=4, c='#2B94E9')
    ax2.set_ylabel('Predicted')

    plt.xlabel('Measured')
    plt.show()

    #残差曲线
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    plt.title('Plotting residuals using training (blue) and test (green) data')
    ax1.scatter(y_pred1, (y_pred1-y_true1)/y_true1,c='#2B94E9',s=40,alpha=0.5)
    ax1.hlines(y=0, xmin=0, xmax=100)
    ax1.set_title("sql_new")
    ax1.set_ylabel('Residuals')

    ax2.scatter(y_pred1_cmp, (y_pred1_cmp-y_true1_cmp)/y_true1_cmp,c='#2B94E9',s=40,alpha=0.5)
    ax2.hlines(y=0, xmin=0, xmax=100)
    ax2.set_title("sql_his")
    ax2.set_ylabel('Residuals')

    plt.xlim([y_true1.min(), y_true1.max()])
    plt.ylim([-4,4])  
    plt.show()





if __name__ == '__main__':
    main()
