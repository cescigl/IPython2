# -*- coding: UTF-8 -*-
import numpy as np
from mysqlpool import Mysql
import datetime


mysql = Mysql()

#90
t_critical = [6.314,2.92,2.353,2.132,2.015,1.943,1.895,1.86,1.833,1.812,1.796,1.782,1.771,1.761,1.753,1.746,1.74,1.734,1.729,1.725,1.721,1.717,1.714,1.711,1.708,1.706,1.703,1.701,1.699,1.697,1.696,1.694,1.692,1.691,1.69,1.688,1.687,1.686,1.685,1.684,1.683,1.682,1.681,1.68,1.679,1.679,1.678,1.677,1.677,1.676,1.675,1.675,1.674,1.674,1.673,1.673,1.672,1.672,1.671,1.671,1.67,1.67,1.669,1.669,1.669,1.668,1.668,1.668,1.667,1.667,1.667,1.666,1.666,1.666,1.665,1.665,1.665,1.665,1.664,1.664,1.664,1.664,1.663,1.663,1.663,1.663,1.663,1.662,1.662,1.662,1.662,1.662,1.661,1.661,1.661,1.661,1.661,1.661,1.66,1.66,1.645]

#80
t_critical = [3.078,1.886,1.638,1.533,1.476,1.44,1.415,1.397,1.383,1.372,1.363,1.356,1.35,1.345,1.341,1.337,1.333,1.33,1.328,1.325,1.323,1.321,1.319,1.318,1.316,1.315,1.314,1.313,1.311,1.31,1.309,1.309,1.308,1.307,1.306,1.306,1.305,1.304,1.304,1.303,1.303,1.302,1.302,1.301,1.301,1.3,1.3,1.299,1.299,1.299,1.298,1.298,1.298,1.297,1.297,1.297,1.297,1.296,1.296,1.296,1.296,1.295,1.295,1.295,1.295,1.295,1.294,1.294,1.294,1.294,1.294,1.293,1.293,1.293,1.293,1.293,1.293,1.292,1.292,1.292,1.292,1.292,1.292,1.292,1.292,1.291,1.291,1.291,1.291,1.291,1.291,1.291,1.291,1.291,1.291,1.29,1.29,1.29,1.29,1.29,1.282]

#50
t_critical = [1,0.816,0.765,0.741,0.727,0.718,0.711,0.706,0.703,0.7,0.697,0.695,0.694,0.692,0.691,0.69,0.689,0.688,0.688,0.687,0.686,0.686,0.685,0.685,0.684,0.684,0.684,0.683,0.683,0.683,0.683,0.683,0.683,0.683,0.683,0.683,0.683,0.683,0.683,0.681,0.681,0.681,0.681,0.681,0.681,0.681,0.681,0.681,0.681,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.679,0.678,0.678,0.678,0.678,0.678,0.678,0.678,0.678,0.678,0.678,0.678,0.678,0.678,0.678,0.678,0.678,0.678,0.678,0.678,0.678,0.677,0.674]
def getPlusDay(day, plus):
    date = datetime.datetime.strptime(str(day),"%Y%m%d") + datetime.timedelta(days=plus)
    string = date.strftime("%Y%m%d")
    return string

def getData(sql):
    """
    获取前一天的原始数据和预测数据
    """
    raw = mysql.getAll(sql)
    return raw


def calHypothesis(l):
    num, retain, predict = l
    #print l
    u = float(retain)*1.0/num
    sigma = (u*(1-u))**0.5
    SE = sigma/(num**0.5)

    t= t_critical[-1] if num>100 else t_critical[0] #
 
    predict_upper_bound = u + t * SE
    predict_lower_bound = u - t * SE

    return (num, float(retain), predict, num*predict_upper_bound, max(num*predict_lower_bound,0.0))

def analysis(result):
    cnt_all = len(result)
    cnt = 0
    for l in result:
        if float(l[2])<=l[4] and float(l[2])>=l[5]:
            cnt += 1

    return cnt_all, cnt

def main():
    today = datetime.date.today().strftime("%Y%m%d")
    yesterday = getPlusDay(today, -1)
    daysAgo3 = getPlusDay(today, -3)
    daysAgo60 = getPlusDay(today, -60)
    daysAgo90 = getPlusDay(today, -90)

    sql = "select a.retain as ori,b.retain,b.retain_predict from (select * from ROI_Retained_Predict_GroupBy_c where date=retain_day and retain>0 and date>="+daysAgo60+") a ,(select * from ROI_Retained_Predict_GroupBy_c where retain>0 and retain_predict>0 and retain_day="+yesterday+") b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.retain>=b.retain and a.retain>=50;"

    sql = "select a.retain as ori,b.retain,b.retain_predict from (select * from ROI_Retained_Predict_GroupBy_c where date=retain_day and retain>0 and date>=20170310) a ,(select * from ROI_Retained_Predict_GroupBy_c where retain>0 and retain_predict>0 and retain_day=20170508) b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.retain>=b.retain and a.retain>=50;"

    sql = "select a.retain as ori,b.spec_retain_predict,b.spec_retain from (select * from ROI_Retained_Predict_GroupBy_c where date=retain_day and pdtid=600025 and spec_retain>0 and date=20170505 and country_iso='' and channel_category='') a ,(select * from ROI_Retained_Predict_GroupBy_c where date=20170505 and spec_retain_predict>0 and country_iso='' and channel_category='') b where a.date=b.date and a.pdtid=b.pdtid and a.country_iso=b.country_iso and a.channel_category=b.channel_category and a.pid=b.pid and a.spec_retain>=b.spec_retain;"

    raw = getData(sql)
    result = map(calHypothesis, raw)
    print result[:10]
    print result[-10:]
    cnt_all, cnt = analysis(result)
    print cnt_all, cnt

if __name__ == '__main__':
    main()


