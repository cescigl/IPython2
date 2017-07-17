# -*- coding: UTF-8 -*-
import geoip2.database as gd
import numpy as np
import pandas as pd
import random
import ConfigParser
from redshiftpool import RedShift
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from mysqlpool import Mysql
import datetime
import os, sys, argparse


class ROI_Init(object):
    """docstring for RedShift"""
    def __init__(self, today=None):
        #读取配置文件

        self.redshift = RedShift()
        self.mysql = Mysql()

        #初始化开始结束日期，初始化产品id
        self.today = today if today else datetime.date.today().strftime("%Y-%m-%d")
        self.yesterday = (datetime.datetime.strptime(str(self.today),"%Y-%m-%d") + datetime.timedelta(days=-1)).strftime("%Y-%m-%d")
        self.day_before_yesterday = (datetime.datetime.strptime(str(self.today),"%Y-%m-%d") + datetime.timedelta(days=-2)).strftime("%Y-%m-%d")
        
        print self.today
        print self.yesterday

        self.yesterday_day = self.yesterday.replace('-','')
        
        self.begin_date = '2017-06-14'
        self.begin_day = self.begin_date.replace('-','')

        self.spec_cond = self.getSpecCondition()

        #构建数据查询sql

        self.purchase_sql = "select date,revenue_day,product_id,classify_group,classify,country_iso,platform,media_source,c_id,c_name,count(distinct token) as pay_user,sum(revenue) as purchase from (select replace(created_time,'-','') as date,replace(log_date,'-','') as revenue_day,classify_group,classify,country_iso,product_id,platform,token,media_source,nvl(c_id,'') as c_id,nvl(nvl(f.campaign_name,campaign),'') as c_name,revenue from  (select created_time,c.log_date,classify_group,classify,c.country_iso,c.product_id,c.platform,token,media_source,c_id,campaign,d.revenue from (select created_time,log_date,classify_group,classify,country_iso,product_id,platform,a.token,media_source,nvl(fb_campaign_id,af_c_id) as c_id,campaign from (select trunc(server_time) as log_date,country_iso,product_id,token,case when sta_value like '%ca-app-pub%' then 'admob' else 'fb' end platform from sta_event_game_publish where server_time>='"+self.day_before_yesterday+"' and server_time<'" + self.yesterday + "' and sta_key in ('_RDA_REWARDOK','_NEW_RDA_REWARDOK')) b , (select distinct created_time,classify_group,classify,t1.gaid,t2.token,media_source,fb_campaign_id,af_c_id,campaign from (select trunc(created_time) as created_time,substring(regexp_substr(grp,'_[^.]*'),len(regexp_substr(grp,'_[^.]*')),1) as classify,substring(regexp_substr(grp,'_[^.]*'),2,len(regexp_substr(grp,'_[^.]*'))-3) as classify_group,split_part(device_info,'#',1) as gaid,split_part(grp,'_',1) as productname from adsdk_abtest_info where created_time>='"+self.begin_date+"' and created_time<'" + self.yesterday + "' and length(split_part(device_info,'#',1))>10 ) t1 ,(select trunc(install_time) as install_time,token,product_id,advertising_id,media_source,fb_campaign_id,af_c_id,campaign from view_appsflyer_postback_android_install_token where install_time>='"+self.begin_date+"' and install_time<'" + self.yesterday + "' and advertising_id is not null) t2,product p where t1.created_time=t2.install_time and t1.gaid=t2.advertising_id and t1.productname=p.pkg and p.pid = t2.product_id) a where b.token=a.token) c left outer join (select log_date,country_iso,product_id,platform,revenue/cnt_all as revenue from (select log_date,country_iso,product_id,platform,count(*) as cnt_all from (select trunc(server_time) as log_date,country_iso,product_id,token,case when sta_value like '%ca-app-pub%' then 'admob' else 'fb' end platform from sta_event_game_publish where server_time>='"+self.day_before_yesterday+"' and server_time<'" + self.yesterday + "' and sta_key in ('_RDA_REWARDOK','_NEW_RDA_REWARDOK') )t group by log_date,country_iso,product_id,platform) a, (select pid,country,revenue,day,channel,ad_type from game_roi_ad_revenue where day='"+self.day_before_yesterday+"' and revenue>0) b where a.log_date=b.day and a.product_id=b.pid and a.country_iso=b.country and a.platform=b.channel) d on c.log_date=d.log_date and c.product_id=d.product_id and c.country_iso=d.country_iso and c.platform=d.platform) e left outer join (select date,campaign_id,campaign_name from view_googleadwords_int where date>=" + self.begin_day + " group by date,campaign_id,campaign_name) f on e.c_id=f.campaign_id and replace(e.created_time,'-','')=f.date) h group by date,revenue_day,classify_group,classify,country_iso,product_id,platform,media_source,c_id,c_name;"

        self.purchase_sql_insert = "insert into ROI_Purchase_ABtest(date,revenue_day,pdtid,classify_group,classify,country_iso,platform,pid,campaign_id,c,pay_user,purchase) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE pay_user=values(pay_user),purchase=values(purchase);"

        self.retain_sql = "select date,'"+self.yesterday_day+"',product_id,classify_group,classify,country_iso,media_source,c_id,campaign,retain_user from (select date,product_id,classify_group,classify,country_iso,media_source,c_id,campaign,count(distinct token) as retain_user from (select replace(created_time,'-','') as date,a.product_id,classify_group,classify,a.country_iso,media_source,nvl(c_id,'') as c_id,nvl(campaign,'') as campaign,b.token from (select created_time,classify_group,classify,product_id,country_iso,t1.gaid,t2.token,media_source,nvl(fb_campaign_id,af_c_id) as c_id,campaign from (select distinct trunc(created_time) as created_time,substring(regexp_substr(grp,'_[^.]*'),len(regexp_substr(grp,'_[^.]*')),1) as classify,substring(regexp_substr(grp,'_[^.]*'),2,len(regexp_substr(grp,'_[^.]*'))-3) as classify_group, split_part(device_info,'#',1) as gaid,split_part(grp,'_',1) as productname from adsdk_abtest_info where created_time>='"+self.begin_date+"' and created_time<'" + self.today + "' and length(split_part(device_info,'#',1))>10 ) t1, (select trunc(install_time) as install_time,token,product_id,country_iso,advertising_id,media_source,fb_campaign_id,af_c_id,campaign from view_appsflyer_postback_android_install_token where install_time>='"+self.begin_date+"' and install_time<'" + self.today + "' and advertising_id is not null)t2, product p where t1.created_time = t2.install_time and t1.gaid=t2.advertising_id and t1.productname=p.pkg and p.pid = t2.product_id) a left join (select distinct trunc(server_time) as server_time,country_iso,token,product_id from sta_event_game_publish where server_time>='" + self.yesterday + "' and server_time<'" + self.today + "' and (product_id=600020 and sta_key not in ('T0I','T0J','_CI') and sta_key not like '_NEW%')) b on a.token = b.token and a.product_id=b.product_id) c group by date,product_id,classify_group,classify,country_iso,media_source,c_id,campaign) d where retain_user>0;"


        self.retain_sql_insert = "insert into ROI_Retained_ABtest(date,retain_day,pdtid,classify_group,classify,country_iso,pid,campaign_id,c,spec_retain) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE spec_retain=values(spec_retain);"


        self.install_sql = "select a.date,product_id,classify_group,classify,country_iso,media_source,nvl(c_id,''),nvl(nvl(f.campaign_name,campaign),'') as c_name,install_num from (select replace(created_time,'-','') as date,product_id,classify_group,classify,country_iso,media_source,nvl(fb_campaign_id,af_c_id) as c_id,campaign,count(distinct t2.token) as install_num from (select distinct trunc(created_time) as created_time,substring(regexp_substr(grp,'_[^.]*'),len(regexp_substr(grp,'_[^.]*')),1) as classify,substring(regexp_substr(grp,'_[^.]*'),2,len(regexp_substr(grp,'_[^.]*'))-3) as classify_group,split_part(device_info,'#',1) as gaid,split_part(grp,'_',1) as productname from adsdk_abtest_info where created_time>='" + self.yesterday + "' and created_time<'" + self.today + "' and length(split_part(device_info,'#',1))>10 ) t1,(select trunc(install_time) as install_time,token,product_id,country_iso,advertising_id,media_source,fb_campaign_id,af_c_id,campaign from view_appsflyer_postback_android_install_token where install_time>='" + self.yesterday + "' and install_time<'" + self.today + "' and advertising_id is not null)t2,product p where t1.gaid=t2.advertising_id and t1.productname=p.pkg and p.pid = t2.product_id group by created_time,product_id,classify_group,classify,country_iso,media_source,nvl(fb_campaign_id,af_c_id),campaign) a left outer join (select date,campaign_id,campaign_name from view_googleadwords_int where date="+self.yesterday_day+" group by date,campaign_id,campaign_name) f on a.c_id=f.campaign_id and a.date=f.date;"

        self.install_sql_insert = "insert into ROI_install_ABtest(date,pdtid,classify_group,classify,country_iso,pid,campaign_id,c,install_num) values(%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE install_num=values(install_num)"



    def getPlusDay(self, day, plus):
        date = datetime.datetime.strptime(str(day),"%Y-%m-%d") + datetime.timedelta(days=plus)
        string = date.strftime("%Y%m%d")
        return string

    def getPlusDate(self, date, plus):
        return (datetime.datetime.strptime(str(date),"%Y-%m-%d") + datetime.timedelta(days=plus)).strftime("%Y-%m-%d")

    def getSpecCondition(self):
        sqlNotIn = "select product_id,not_in_condition from spec_condition where in_condition=''"
        sqlIn = "select product_id,in_condition from spec_condition where not_in_condition=''"
        temp = self.getSpecConditionDetail(sqlNotIn, ' and sta_key not in ')
        cond = temp
        temp = self.getSpecConditionDetail(sqlIn, ' and sta_key in ')
        cond = cond + temp
        cond = cond.strip(' or ')
        return cond


    def getSpecConditionDetail(self, sql, keystr):
        cond = ''
        rawNotIn = self.mysql.getAll(sql)
        #print rawNotIn
        for r in rawNotIn:
            temp = '(product_id=' + str(r[0]) + keystr + str(r[1]) + ')'
            #print temp
            cond = cond + temp + ' or '
        return cond 


    def insertData(self, roisql, insertsql):
        print roisql
        raw = self.redshift.getAll(roisql)
        print raw
        print insertsql
        self.mysql.insertMany(insertsql,raw)



    def analysis(self):
        self.insertData(self.purchase_sql, self.purchase_sql_insert)
        self.insertData(self.retain_sql, self.retain_sql_insert)
        self.insertData(self.install_sql, self.install_sql_insert)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--today', type=str, help='日期')
    args = parser.parse_args()
    print("[args]: ", args)

    ri = ROI_Init(args.today)
    result = ri.analysis()

