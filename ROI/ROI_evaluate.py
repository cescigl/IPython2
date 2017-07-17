# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import datetime
import MySQLdb
import time
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


class Evaluate(object):
    """docstring for Evaluate"""
    def __init__(self):
        super(Evaluate, self).__init__()
        
        self.today = datetime.date.today().strftime("%Y%m%d")
        self.date_3day_ago = (datetime.date.today() - datetime.timedelta(days=3)).strftime("%Y%m%d")
        self.date_7day_ago = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y%m%d")
        self.makeTimeList(self.date_7day_ago, self.date_3day_ago)

        self.db = baseModel().conn()
        self.cursor = self.db.cursor()

        self.f=open('log/evaluate.log','a')


        self.sql_retain = "select date,retain,retain_predict from ROI_Retained_Predict_GroupBy_c where date>=" + self.date_7day_ago + " and date<=" + self.date_3day_ago + " and pdtid in (600001,600004,600007,600008,600018,600020,600022,600025,600027,600029,600030) and retain>10 and retain_predict>0;"
        
        self.sql_spec_retain = "select date,spec_retain,spec_retain_predict from ROI_Retained_Predict_GroupBy_c where date>=" +self.date_7day_ago + " and date<=" + self.date_3day_ago + " and pdtid in (600001,600004,600007,600008,600018,600020,600022,600025,600027,600029,600030) and spec_retain>10 and spec_retain_predict>0;"
        
        self.sql_imp_num = "select revenue_date,imp_num,imp_num_predict from ROI_News_Predict_GroupBy_c where revenue_date>=" + self.date_7day_ago + " and revenue_date<=" + self.date_3day_ago + " and pdtid in (600001,600004,600007,600008,600018,600020,600022,600025,600027,600029,600030) and imp_num>10 and imp_num_predict>0;"

        self.sql_click_num = "select revenue_date,click_num,click_num_predict from ROI_News_Predict_GroupBy_c where revenue_date>=" + self.date_7day_ago + " and revenue_date<=" + self.date_3day_ago + " and pdtid in (600001,600004,600007,600008,600018,600020,600022,600025,600027,600029,600030) and click_num>10 and click_num_predict>0;"

        self.sql_imp_revenue = "select revenue_date,imp_revenue,imp_revenue_predict from ROI_News_Predict_GroupBy_c where revenue_date>=" + self.date_7day_ago + " and revenue_date<=" + self.date_3day_ago + " and pdtid in (600001,600004,600007,600008,600018,600020,600022,600025,600027,600029,600030) and imp_revenue>10 and imp_revenue_predict>0;"

        self.sql_click_revenue = "select revenue_date,click_revenue,click_revenue_predict from ROI_News_Predict_GroupBy_c where revenue_date>=" + self.date_7day_ago + " and revenue_date<=" + self.date_3day_ago + " and pdtid in (600001,600004,600007,600008,600018,600020,600022,600025,600027,600029,600030) and click_revenue>10 and click_revenue_predict>0;"

        self.sql_purchase = "select revenue_day,purchase,purchase_predict from ROI_Purchase_Predict_GroupBy_c where  revenue_day>=" + self.date_7day_ago + " and revenue_day<=" + self.date_3day_ago + " and pdtid in (600001,600004,600007,600008,600018,600020,600022,600025,600027,600029,600030) and purchase>10 and purchase_predict>0;"
        

    def makeTimeList(self, begin, end):
        self.timeList = [begin]
        plus = 1
        while(True):
            date = datetime.datetime.strptime(str(begin),"%Y%m%d") + datetime.timedelta(days=plus)
            string = date.strftime("%Y%m%d")
            plus = plus + 1 
            self.timeList.append(string)
            if string == end:
                break
        print self.timeList
        

    def getData(self, sql):
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        data = np.array(data)
        return data

    def saveFile(self, l):
        result = ''
        for i in range(len(l)-1):
            result = result + str(l[i]) + '|'
        result = result + str(l[len(l)-1]) + '\n'

        self.f.write(result)



    def evaluate(self, sql, key):
        data = self.getData(sql)

        for times in self.timeList:
            loc = np.where(data[:,0]==times)
            tmp = data[loc]
            #print tmp
            y_true = tmp[:,1].astype('float32')
            y_pred = tmp[:,2].astype('float32')

            r2 = r2_score(y_true, y_pred)
            mean_se = mean_squared_error(y_true, y_pred)
            mean_ae = mean_absolute_error(y_true, y_pred)
            median_ae = median_absolute_error(y_true, y_pred)

            print key,times,r2,mean_ae,median_ae
            l=[key,times,r2,mean_ae,median_ae]
            self.saveFile(l)

    def close(self):
        self.f.close()
        self.cursor.close()

    def main(self):
        self.evaluate(self.sql_retain, 'Retain')
        self.evaluate(self.sql_spec_retain, 'Spec_Retain')
        self.evaluate(self.sql_imp_num, 'imp_num')
        self.evaluate(self.sql_click_num, 'click_num')
        self.evaluate(self.sql_imp_revenue, 'imp_revenue')
        self.evaluate(self.sql_click_revenue, 'click_revenue')
        self.evaluate(self.sql_purchase, 'purchase')
        self.close()


if __name__ == '__main__':
    eva = Evaluate()
    eva.main()
