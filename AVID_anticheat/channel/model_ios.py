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


class ActiveFraud(object):
    """docstring for RedShift"""
    def __init__(self, ptid=None, begin=None, end=None):
        #读取配置文件
        self.reader = gd.Reader('GeoIP2-City.mmdb')
        self.cf = ConfigParser.ConfigParser()
        self.cf.read('imsiConfig2.conf')

        self.redshift = RedShift()
        self.mysql = Mysql()

        #初始化开始结束日期，初始化产品id
        self.today = end if end else datetime.date.today().strftime("%Y-%m-%d")
        self.yesterday = begin if begin else (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        self.before_yesterday = (datetime.datetime.strptime(str(self.yesterday),"%Y-%m-%d") + datetime.timedelta(days=-1)).strftime("%Y-%m-%d")
        self.date_7day_ago = (datetime.datetime.strptime(str(self.yesterday),"%Y-%m-%d") + datetime.timedelta(days=-7)).strftime("%Y-%m-%d")
        self.ptid = ptid if ptid else '600025'
        print self.today
        print self.yesterday
        print self.ptid

        self.day_yesterday = self.yesterday.replace('-','') #20100101
        self.day_before_yesterday = self.getPlusDay(self.yesterday, -1) #20100101
        self.day_7day_ago = self.getPlusDay(self.yesterday, -7) #20100101

        #构建数据查询sql
        self.sql_channel_cnt = "select media_source,count(distinct token) from view_appsflyer_postback_ios_install_token where install_time>='" + self.yesterday + "' and install_time<'" + self.today + "' and product_id=" + self.ptid + " and token is not null group by media_source;"
        self.sql_channel_ip_cnt = "select media_source,sum(cnt) from (select media_source,ip,count(distinct token) as cnt from view_appsflyer_postback_ios_install_token where install_time>='" + self.yesterday + "' and install_time<'" + self.today + "' and product_id=" + self.ptid + " and token is not null  group by media_source,ip having count(distinct token)>=5) group by media_source;"
        self.sql_channel_ip_imsi = "select media_source,country_iso,city,carrier,ip,split_part(imsi,'_',2) as imsi from view_appsflyer_postback_ios_install_token where install_time>='" + self.yesterday + "' and install_time<'" + self.today + "' and product_id=" + self.ptid + " and token is not null ;"
        self.sql_channel_imsi_cnt = "select media_source,sum(case when lenimsi=15 then 1 else 0 end) as cnt from (select distinct media_source,token,len(split_part(imsi,'_',2)) as lenimsi from view_appsflyer_postback_ios_install_token where install_time>='" + self.yesterday + "' and install_time<'" + self.today + "' and product_id=" + self.ptid + " and token is not null ) t group by media_source;"
        self.sql_channel_imei_cnt = "select media_source,sum(case when lenimei=15 then 1 else 0 end) as cnt from (select distinct media_source,token,len(imei) as lenimei from view_appsflyer_postback_ios_install_token where install_time>='" + self.yesterday + "' and install_time<'" + self.today + "' and product_id=" + self.ptid + " and token is not null ) t group by media_source;"
        self.sql_channel_hour_cnt = "select media_source,date_part(hour, install_time),count(distinct token) from view_appsflyer_postback_ios_install_token where install_time>='" + self.yesterday + "' and install_time<'" + self.today + "' and product_id=" + self.ptid + " and token is not null  group by media_source,date_part(hour, install_time);"
        self.sql_channel_wifi_cnt = "select media_source,sum(case when wifi='t' then cnt else 0 end) as cnt from (select media_source,wifi,count(distinct token) as cnt from view_appsflyer_postback_ios_install_token where install_time>='" + self.yesterday + "' and install_time<'" + self.today + "' and product_id=" + self.ptid + " and token is not null group by media_source,wifi) t group by media_source;"
        self.sql_channel_ver_cnt = "select media_source,ver,count(distinct token) from view_appsflyer_postback_ios_install_token where install_time>='" + self.yesterday + "' and install_time<'" + self.today + "' and product_id=" + self.ptid + " and token is not null  group by media_source,ver;"
        self.sql_channel_device_cnt = "select media_source,device_brand,count(distinct token) from view_appsflyer_postback_ios_install_token where install_time>='" + self.yesterday + "' and install_time<'" + self.today + "' and product_id=" + self.ptid + " and token is not null  group by media_source,device_brand;"
        self.sql_channel_IPseg_cnt = "select media_source,sum(cnt) from (select media_source,ip,count(token) as cnt from (select distinct media_source,token,split_part(ip,'.',1)||'.'||split_part(ip,'.',2)||'.'||split_part(ip,'.',3) as ip from view_appsflyer_postback_ios_install_token where install_time>='" + self.yesterday + "' and install_time<'" + self.today + "' and product_id=" + self.ptid + " and token is not null ) t group by media_source,ip having count(token)>=5) t2 group by media_source;"
        self.sql_channel_date = "select distinct media_source,'" + self.yesterday + "','" + self.ptid + "' from view_appsflyer_postback_ios_install_token where install_time>='" + self.yesterday + "' and install_time<'" + self.today + "' and product_id=" + self.ptid + " and token is not null;"
        #构建t分布预测模型
        y = np.array([0.50,0.60,0.70,0.80,0.90,0.95,0.98])
        x = np.array([0.674,0.842,1.036,1.282,1.645,1.960,2.326])
        x = x.reshape(x.shape[0],1)        
        X = self.buildComplexDataset(x)

        alpha = 0.001
        l1_ratio = 0.001
        enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        self.model = enet.fit(X, y)

    def getPlusDay(self, day, plus):
        date = datetime.datetime.strptime(str(day),"%Y-%m-%d") + datetime.timedelta(days=plus)
        string = date.strftime("%Y%m%d")
        return string

    def getPlusDate(self, date, plus):
        return (datetime.datetime.strptime(str(date),"%Y-%m-%d") + datetime.timedelta(days=plus)).strftime("%Y-%m-%d")

    def insertLoyaltyUser(self, ):

        sql_loyalty = "select install_time,media_source,product_id,sum(uncheat) as loyuser,sum(uncheat)*1.0/count(*) as loyrate from (select install_time,product_id,token,media_source, case when cnt_all>=100 then 1 when cnt_bg*1.0/cnt_all>0.4 then 0 else 1 end as uncheat from (select install_time,product_id,token,media_source, sum(cnt_key) as cnt_all, sum(case t3.sta_key when '_' then cnt_key else 0 end) as cnt_bg from  (select trunc(t1.install_time) as install_time,product_id,t1.token,t1.media_source,t2.sta_key,count(*) as cnt_key from (select media_source,product_id,token,install_time from view_appsflyer_postback_ios_install_token where  install_time>='" + self.yesterday + "' and install_time<'" + self.today + "' and product_id=" + self.ptid + ") t1 left outer join (select token,substring(sta_key,1,1) as sta_key,server_time from sta_event_game_publish where product_id=" + self.ptid + " and server_time>='" + self.yesterday + "' and server_time<'" + self.today + "') t2 on t1.token=t2.token group by trunc(t1.install_time),product_id,t1.token,t1.media_source,t2.sta_key) t3  group by install_time,product_id,token,media_source ) t4 ) t5 group by install_time,product_id,media_source;"

        sql_insert = "insert into active_fraud_analysis(day,channel,product_id,loyalty_users,loyalty_users_rate) values(%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE loyalty_users=values(loyalty_users),loyalty_users_rate=values(loyalty_users_rate)"

        raw = self.redshift.getAll(sql_loyalty)
        self.mysql.insertMany(sql_insert,raw)

    def insertRetain(self):
        """
        把隔日留存数据插进去
        """
        sql = "INSERT INTO active_fraud_analysis (day,channel,product_id,retain) select * from (select str_to_date(t1.date, '%Y%m%d') as day,t1.pid,t1.pdtid,(case when t2.cnt is null then 0 else t2.cnt end)*1.0/t1.cnt as rate from (select date,pid,pdtid,sum(match_num) cnt from ROI_install where pdtid=" + self.ptid + " and date='"+self.day_before_yesterday+"' group by pid,pdtid) t1 left outer join (select date,pid,pdtid,retain_day,sum(spec_retain) cnt from ROI_Retained where pdtid=" + self.ptid + " and date='"+self.day_before_yesterday+"' and retain_day='" + self.day_yesterday + "'  group by pid) t2 on t1.date=t2.date and t1.pid=t2.pid) t on DUPLICATE KEY UPDATE retain=rate;"
        #print sql
        self.mysql.insertOne(sql,None)
    
    def insertAvgReturn7day(self):
        sql = "INSERT INTO active_fraud_analysis (day,channel,product_id,avg_return_7day) select * from (select str_to_date(t1.date, '%Y%m%d') as day,t1.pid,t1.pdtid,t2.cnt/(datediff(str_to_date('" + self.day_yesterday + "','%Y%m%d'),str_to_date(t1.date, '%Y%m%d'))*t1.cnt) rate from (select date,pid,pdtid,sum(match_num) cnt from ROI_install where pdtid=" + self.ptid + " and date>='"+self.day_7day_ago+"' and date<'"+self.day_yesterday+"' group by pid,pdtid,date) t1 left outer join (select date,pid,pdtid,sum(cnt) as cnt from (select date,pid,pdtid,retain_day,sum(spec_retain) cnt from ROI_Retained where pdtid=" + self.ptid + " and date>='"+self.day_7day_ago+"' and retain_day>date and retain_day<='" + self.day_yesterday + "' group by pid,retain_day,date) t group by date,pid,pdtid) t2 on t1.date=t2.date and t1.pid=t2.pid) t3 on DUPLICATE KEY UPDATE avg_return_7day=rate;"

        self.mysql.insertOne(sql,None)

    def insertScore(self):

        sql_new = "select day,channel,product_id,ifnull(loyalty_users_rate,0),ifnull(retain,0),ifnull(avg_return_7day,0),ifnull(net_rate,0),ifnull(imsi_rate,0),ifnull(imei_rate,0),ifnull(hour_rate,0),ifnull(device_rate,0) from active_fraud_analysis where install>50 and day>='"+self.date_7day_ago+"' and day<'"+self.today+"' and product_id ="+ self.ptid +" ;"
        sql_insert = "insert into active_fraud_analysis(day,channel,product_id,score) values(%s,%s,%s,%s) ON DUPLICATE KEY UPDATE score=values(score)"

        raw = self.mysql.getAll(sql_new)
        if not raw:
            sys.exit()
        
        for row in raw:
            print row
            # 留存为空的不处理
            if row[4]==0.0:
                return ''
            value = [row[i] for i in range(3)]
            rate = [float(row[i]) for i in range(3,len(row))]
              
            score = reduce(lambda x,y:x*y,np.array(rate)) * 10000 / (row[4]/row[5]) ** 3
            if row[1] == 'Organic':
                score = 100.0
            
            value.extend([score])
            self.mysql.insertOne(sql_insert,value)
            
        return raw

    def modifyScore(self):
        sql = "update active_fraud_analysis a, (select t2.day,t2.channel,t2.product_id,score*90/benchmark as score from (select day,product_id,max(score) as benchmark from active_fraud_analysis where install>50 and channel<>'Organic' and score>0 and product_id="+ self.ptid +" and day>='"+self.date_7day_ago+"' group by day,product_id) t1,(select day,channel,product_id,score from active_fraud_analysis where install>50 and channel<>'Organic' and score>0 and product_id="+ self.ptid +" and day>='"+self.date_7day_ago+"') t2 where t1.day=t2.day  and t1.product_id=t2.product_id) b set a.score=b.score where a.day=b.day  and a.product_id=b.product_id and a.channel=b.channel;"
        self.mysql.update(sql)

    def sigmoid(self, X, useStatus):  
        """
        归一化处理，把比率数据归一化到[0.5,1]
        """
        if useStatus:  
            return 1.0 / (1 + np.exp(-(X)));  
        else:  
            return X;  

    def makeDF(self, dt, ind=False, col=['key', 'cnt']):
        """
        数据以列表形式保存，其中的元素为元祖  [('a',1),('b',2)]
        """
        if ind:
            indexs = [i[0] for i in dt]
            data = [i[1:] for i in dt]
            df = pd.DataFrame(data, index=indexs, columns=col)
        else:
            df = pd.DataFrame(dt, columns=col)
        return df

    def analysisIP_IMSI(self):
        """
        分析IP和IMSI的一致性，获取原始数据
        """
        raw = self.redshift.getAll(self.sql_channel_ip_imsi)
        result = map(self.judgeIP_IMSI, raw) 
        a=[i for i in result if i[1]==0]
        b=[i for i in result if i[1]==1]
        dfa = self.makeDF(a, False, ['key','cnt_IPIMSI'])
        c = dfa.groupby('key').count()
        #dfb = self.makeDF(b, False, ['key','cntIPIMSI1'])
        #d = dfb.groupby('key').count()
        df = pd.concat([c], axis=1)

        return df

    def judgeIP_IMSI(self, l):
        """
        分析IMSI MCC和MNC白名单，验证其和IP国家的一致性。
        """
        media_source,country_iso,city,carrier,ip,imsi = l
        if not imsi or len(imsi)<15:
            return (media_source, 0)
        try:
            response = self.reader.city(ip)
            #英文名
            ipiso = response.country.iso_code #TW
            mcc = imsi[0:3]
            if self.cf.has_option('DB',mcc):
                value = self.cf.get('DB',mcc)
                countryList = []
                mncList = value.split(',')[1:]
                if imsi[3:5] in mncList or imsi[3:6] in mncList:
                    countryList = value.split(',')[0].split('#')
                    if country_iso == ipiso and ipiso in countryList:
                        return (media_source, 1)
        except Exception as err:
            print err
            return (media_source, 0)
        return (media_source, 0)

    def buildComplexDataset(self, x):
        temp = 1.0/x
        X = np.hstack((x,temp))
        return X

    def confidenceIntervalEstimation(self, l):
        """
        根据待评估数据的结果，评估其距organic的距离，计算器可信度。
        """
        media_source, n, u0, u, tag, rate = l 
        if n<50:
            #print [media_source, None]
            return [media_source, 0.01]
        if tag and u<=u0:
            return [media_source, 1.0]
        if u0 == u:
            return [media_source, 1.0]
        sigm = (u0*(1-u0)) ** rate
        se = sigm / (n ** rate)

        x = abs(u-u0)/(se+random.random()/10)

        X_predict = np.array([x])
        X_predict = X_predict.reshape(X_predict.shape[0],1)
        X_predict = self.buildComplexDataset(X_predict)
        
        pred = self.model.predict(X_predict)[0]
        result = min(1,max(0.01,1-pred))

        #print [media_source, result]
        return [media_source, result]

    def analysisHour(self):
        """
        统计各渠道各小时的激活数目，以organic为基准，计算人数最多的三个小时占比，统计其他渠道对应3小时的比例，用置信区间95%来衡量渠道的真实性。
        """
        raw = self.redshift.getAll(self.sql_channel_hour_cnt)
        #排序选出最大的三个小时
        print raw
        print self.sql_channel_hour_cnt
        if not raw:
            sys.exit()
        rawOrganic = [i for i in raw if i[0] == 'Organic']
        rawOrganic.sort(key=lambda x:-x[2])
        print rawOrganic


        organicMost = [rawOrganic[i][1] for i in range(min(3,len(rawOrganic)))]
        #分别求总数和前三个月聚合累加
        raw3most = [(i[0],i[2]) for i in raw if i[1] in organicMost]
        rawAll = [(i[0],i[2]) for i in raw]

        df3most = self.makeDF(raw3most, False, ['key1', 'cnt_hour'])
        df3mostSum = df3most.groupby('key1').sum()
        #dfAll = self.makeDF(rawAll, False, ['key2', 'cnt_hour_all'])
        #dfAllSum = dfAll.groupby('key2').sum()

        #计算出联合dataframe
        result = pd.concat([df3mostSum], axis=1)
        return result

    def analysisDevice(self):
        """
        分析设备分布
        """
        raw = self.redshift.getAll(self.sql_channel_device_cnt)
        if not raw:
            sys.exit()
        rawOrganic = [i for i in raw if i[0] == 'Organic']
        rawOrganic.sort(key=lambda x:-x[2])
        organicMost = [rawOrganic[i][1] for i in range(min(3,len(rawOrganic)))]
        #分别求总数和前三个月聚合累加
        raw3most = [(i[0],i[2]) for i in raw if i[1] in organicMost]
        rawAll = [(i[0],i[2]) for i in raw]

        df3most = self.makeDF(raw3most, False, ['key1', 'cnt_device'])
        df3mostSum = df3most.groupby('key1').sum()
        #dfAll = self.makeDF(rawAll, False, ['key2', 'cnt_device_all'])
        #dfAllSum = dfAll.groupby('key2').sum()

        #计算出联合dataframe
        result = pd.concat([df3mostSum], axis=1)
        return result

    def analysisModel(self, sql, index, col):
        #print sql
        raw = self.redshift.getAll(sql)
        print raw
        df = self.makeDF(raw, index, col)
        return df

    def makeConfidenceIntervalEstimation(self, df, col, tag=False, rate=0.3):
        """
        tag表示是否是单边检测，还是中心检测，如果为True，且是单边检测，检测数据比率小于organic数据，可信度为1。默认是False
        """
        dfData = df.fillna(0).reset_index().values
        u0 = [i[2]*1.0/i[1] for i in dfData if i[0] == 'Organic'][0]
        dfList = [[i[0], i[1], u0, i[2]*1.0/i[1], tag, rate] for i in dfData]
        result = map(self.confidenceIntervalEstimation, dfList)
        dfResult = self.makeDF(result, True, col) 
        
        return dfResult

    def rateEstimation(self, l):
        media_source, n, u0, u, tag, rate = l 
        print l

        if u>=u0:
            return [media_source, max((2*u0-u)/(u0+0.01),0.1)]
        else:
            return [media_source, u/(u0+0.01)]

    def makeConfidenceRate(self, df, col, tag=False, rate=0.3):
        """
        tag表示是否是单边检测，还是中心检测，如果为True，且是单边检测，检测数据比率小于organic数据，可信度为1。默认是False
        """
        dfData = df.fillna(0).reset_index().values
        u0 = [i[2]*1.0/i[1] for i in dfData if i[0] == 'Organic'][0]

        dfList = [[i[0], i[1], u0, i[2]*1.0/i[1], tag, rate] for i in dfData]
        result = map(self.rateEstimation, dfList)
        dfResult = self.makeDF(result, True, col) 
        return dfResult

    def saveToMysql(self, df):
        data = df.fillna(0).reset_index().values
        sql = "insert into active_fraud_analysis(channel,day,product_id,install,cnt_ip,ip_rate,cnt_net,net_rate,cnt_imsi,imsi_rate,cnt_imei,imei_rate,cnt_hour,hour_rate,cnt_device,device_rate) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE install=values(install),cnt_ip=values(cnt_ip),ip_rate=values(ip_rate),cnt_net=values(cnt_net),net_rate=values(net_rate),cnt_imsi=values(cnt_imsi),imsi_rate=values(imsi_rate),cnt_imei=values(cnt_imei),imei_rate=values(imei_rate),cnt_hour=values(cnt_hour),hour_rate=values(hour_rate),cnt_device=values(cnt_device),device_rate=values(device_rate)"

        #print data
        values =[(i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9],i[10],i[11],i[12],i[13],i[14],i[15]) for i in data]
        print "values"
        print values
        self.mysql.insertMany(sql,values)
        #from sqlalchemy import create_engine  
        ##将数据写入mysql的数据库，但需要先通过sqlalchemy.create_engine建立连接,且字符编码设置为utf8，否则有些latin字符不能处理  
        #yconnect = create_engine('mysql+mysqldb://datauser:ebOUeWSin3hAAKKD@test-dataverse-web.c0poh9vgjxya.rds.cn-north-1.amazonaws.com.cn:3306/dataverse?charset=utf8')  
       
        #pd.io.sql.to_sql(df, 'zhx_test', yconnect, flavor='mysql', schema='dataverse', if_exists='append')
        #pd.io.sql.to_sql(rawChannel,'test_zhx', yconnect, flavor='mysql', schema='dataverse', if_exists='append')


    def analysis(self):
        
        print 'BEGIN'
        rawData = self.analysisModel(self.sql_channel_date, True, ['datetime','product_id']) # 日期
        print 'rawData'
        rawChannel = self.analysisModel(self.sql_channel_cnt, True, ['channel']) #总数
        print 'rawChannel'
        rawIP = self.analysisModel(self.sql_channel_ip_cnt, True, ['cnt_IP'])    #IP重复分布
        print 'rawIP'
        rawNetwork = self.analysisModel(self.sql_channel_wifi_cnt, True, ['cnt_Net'])    #网络分布
        print 'rawNetwork'
        #rawIPseg = self.analysisModel(self.sql_channel_IPseg_cnt, True, ['cnt_IPseg'])    #IP段分布
        rawImsi = self.analysisModel(self.sql_channel_imsi_cnt, True, ['cnt_IMSI']) #IMSI为空比率
        print 'rawImsi'
        rawImei = self.analysisModel(self.sql_channel_imei_cnt, True, ['cnt_IMEI']) #IMEI为空比率
        print 'rawImei'
        #rawIpImsi = self.analysisIP_IMSI()  #IP、IMSI异常分布
        
        rawHour = self.analysisHour() #时间分布
        print 'rawHour'
        rawDevice = self.analysisDevice() #设备分布

        
        device = pd.concat([rawChannel, rawDevice], axis=1)
        
        #dfDevice = self.makeConfidenceIntervalEstimation(device, ['device_rate'],rate=0.5)
        dfDevice = self.makeConfidenceRate(device, ['device_rate'],rate=0.5)
        ip = pd.concat([rawChannel, rawIP], axis=1)
        dfIP = self.makeConfidenceIntervalEstimation(ip, ['ip_rate'], True)

        netWork = pd.concat([rawChannel, rawNetwork], axis=1)
        #dfNetwork = self.makeConfidenceIntervalEstimation(netWork, ['network_rate'],rate=0.3)
        dfNetwork = self.makeConfidenceRate(netWork, ['network_rate'],rate=0.3)

        #IPseg = pd.concat([rawChannel, rawIPseg], axis=1)
        #dfIPseg = self.makeConfidenceIntervalEstimation(IPseg, ['ipseg_rate'])
        
        hour = pd.concat([rawChannel, rawHour], axis=1)
        #dfHour = self.makeConfidenceIntervalEstimation(hour, ['hour_rate'], rate=0.5)
        dfHour = self.makeConfidenceRate(hour, ['hour_rate'], rate=0.5)
        #IpImsi = pd.concat([rawChannel, rawIpImsi], axis=1)
        #dfIpImsi = self.makeConfidenceIntervalEstimation(IpImsi, ['IpImsi_rate'])

        imsi = pd.concat([rawChannel, rawImsi], axis=1)
        dfImsi = self.makeConfidenceRate(imsi, ['imsi_rate'], rate=0.3)

        imei = pd.concat([rawChannel, rawImei], axis=1)
        dfImei = self.makeConfidenceRate(imei, ['imei_rate'], rate=0.3)

        result = pd.concat([rawData, rawChannel, rawIP, dfIP, rawNetwork, dfNetwork, rawImsi, dfImsi, rawImei, dfImei, rawHour, dfHour, rawDevice, dfDevice], axis=1)
        print result
        print 'result'
        self.saveToMysql(result)
        self.insertRetain()
        self.insertAvgReturn7day()
        self.insertLoyaltyUser()
        
        result = self.insertScore()
        self.modifyScore()
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--product_id', type=str, help='产品id')
    parser.add_argument('--begin', type=str, help='开始日期')
    parser.add_argument('--end', type=str, help='结束日期')
    args = parser.parse_args()
    print("[args]: ", args)

    af = ActiveFraud(args.product_id, args.begin, args.end)
    result = af.analysis()
    #af.insertScore()
    #af.modifyScore()
