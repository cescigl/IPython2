# -*- coding: utf-8 -*-
from flask import Flask
from flask_restful import Resource, Api, reqparse, fields, marshal_with
import time
import random 
import json
import numpy as np
import sys
sys.path.append("/data/mesh_push_service/meshapi/")
from common.redispool import RedisCache
#from common.ddbpool import DDB
import logging    
import os  
from logging.handlers import TimedRotatingFileHandler 
  

class MeshRecommend(Resource):
    def get(self, arg):
        # arg =  "mesh_id=123&tag=456"
        print "###### Begin Rec ##########"
        begintime = int(time.time()*1000)
        print arg
        mesh_id = arg.split("&")[0].split('=')[1]
        tag = arg.split("&")[1].split('=')[1]
        ver = arg.split("&")[2].split('=')[1]

        rec = Recommend(mesh_id, tag, ver)
        result = rec.rec()
        endtime = int(time.time()*1000)
        if endtime - begintime >800:
            print mesh_id + " REC TIMEOUT. " + str(endtime - begintime)
        print "REC RESULT " + str(result)
        print "###### End Rec ##########"
        return {'result':'#'.join(result)} # 用#拼接的字符串。

class Recommend(object):
    """ make Recommend result """
    def __init__(self, mesh_id, tag='', ver=''):
        super(Recommend, self).__init__()

        self.log_filename="/data/mesh_push_service/meshapi/log/mesh_matching_logging.txt" 
        self.log_format=' [%(asctime)s]   %(message)s' 

        #将日志文件格式化  
        import logging
        logging.basicConfig(format=self.log_format,datafmt='%Y-%m-%d %H:%M:%S %p',level=logging.DEBUG,filename=self.log_filename,filemode='a') 

        self.mesh_id = mesh_id #mesh id
        self.tag = tag #匹配暗号
        self.ver = ver #匹配版本

        # 根据版本不同，确定在线用户redis key
        if self.ver <> '':
            if self.tag <> '':
                #String.format("%s%s_%s", tagMatchListKeyPrefix, ver, formattedTag)
                self.onlineUserKey = '_tt_waiting_4_match_list_' + self.ver + '_' + self.tag
                
            else:
                #String.format("%s_%s", matchListKey, ver);
                self.onlineUserKey = '_tt_waiting_4_match_list_' + self.ver
        else:
            if self.tag <> '':
                #String.format("%s%s", tagMatchListKeyPrefix, formattedTag) : matchListKey;
                self.onlineUserKey = '_tt_waiting_4_match_list_' + self.tag
            else:
                self.onlineUserKey = '_tt_waiting_4_match_list'

        print self.onlineUserKey

        self.redis = RedisCache()
        #self.ddb = DDB()
        self.recLen = 10 # 推荐结果的长度

        self.accompanyChatKey = '_tt_fake_user_id_list' #陪聊redis (set)
        #self.onlineUserKey = '_tt_waiting_4_match_list'  #在线用户 redis key，有效数据时4s内活跃数据
        #self.onlineUserTagKeyPrex = '_tt_waiting_4_match_list_'  #使用暗号匹配的在线用户 redis key，有效数据时4s内活跃数据
        self.matchingUserKey = '_tt_locked_user_list' #正在匹配中的用户  redis key，有效数据时10s内活跃数据
        self.userDetail = '_tt_User' #保存用户的所有属性， hset
        self.blockUserPrex = '_tt_user_black_list_' #记录被用户拉黑的用户
        self.reportUserPrex = '_tt_user_report_list_' # 记录被用户举报的用户
        self.haveMeshedPrex = '_tt_temporary_shield_user_list_' #记录用户匹配过的历史, 两个用户是对等的，记录的时候需要记录两条
        self.activityKey = 'LUCKY_TAG' #保存活动的特殊tag

        #gender 0-女性 1-男性 meshgender 101-女性 102-男性
        self.userGenderMaleMale = 'rec_user_gender_male_male' # 保存男性用户愿意匹配男性用户的列表
        self.userGenderMaleFemale = 'rec_user_gender_male_female' # 保存男性用户愿意匹配女性用户的列表
        self.userGenderFemaleMale = 'rec_user_gender_female_male' # 保存女性用户愿意匹配男性用户的列表
        self.userGenderFemaleFemale = 'rec_user_gender_female_female' # 保存女性用户愿意匹配女性用户的列表

        self.newPrex = 'rec_new_match_' #记录匹配过的用户，匹配过即变成老用户
        self.accompanyUserRecSort = 'rec_accompany_user_sort' #记录陪聊用户的最新一次被推荐时间。
        self.userFriendPrex = '_tt_user_id_2_friend_user_id_list_' #记录用户的好友列表 （set）

        self.userLeft_factor = "rec_userLeft_factor" #记录用户特征向量
        self.userRight_factor = "rec_userRight_factor" 
        self.userPopular = 'rec_popular' # 记录用户的流行度 hset
        self.doubtful = '_tt_doubtful_users' #记录涉黄用户和举报3次用户

    def ifNewID(self):
        """ 
        通过用户信息表判断是否是新用户，通过过去5天的用户匹配请求判断是否是新用户，每个用户有5次作为新用户匹配的机会,前两次概率是100%，后三次概率是30%
        """
        ifNew = False
        redisKey = self.newPrex + self.mesh_id

        dic = self.redis.get_data(redisKey)
        
        if dic['result'] is None:
            ifNew = True
            self.redis.set_data(redisKey, 1)
            self.redis.set_expire(redisKey, 86400*5)
            return ifNew
        else:
            num = int(dic['result'])
            self.redis.set_data(redisKey, num + 1)
            self.redis.set_expire(redisKey, 86400*5)
            if num <= 10:
                ifNew = True
                return ifNew
            elif num <= 20:
                ifNew = (True if random.random()<0.3 else False)
                return ifNew

        # 老用户20%的概率以老用户处理
        ifNew = (True if random.random()<0.2 else False)
        return ifNew

    def ifDoubtful(self):
        """
        24小时内被3人以上举报或者 24小时内疑似涉黄
        """
        ct = int(time.time()*1000)
        doubtfulList = self.redis.zset_zrangebyscore(self.doubtful, ct-24*3600*1000, ct)['result']

        if self.mesh_id in doubtfulList:
            #扣减当前用户的评分
            self.redis.hset_hset(self.userPopular, self.mesh_id, 0.0)
            return True
        return False
    
    def luckyTag(self):
        """
        判断用户的tag是否是活动tag
        """
        acTag = self.redis.get_data(self.activityKey)['result']
        lucky = (True if self.tag == acTag else False)
        return lucky

    def listDifference(self, left, right): #已测试
        if left:
            if right:
                return list(set(left).difference(set(right)))
            return left
        return []

    def listIntersection(self, left, right): #已测试
        if left:
            if right:
                return list(set(left).intersection(set(right)))
        return []

    def filterHaveMeshed(self, user, currentTime):
        """
        过滤已经匹配过的用户,5分钟内匹配过的不再匹配。
        """
        haveMeshedWithScore = self.redis.zset_zrangebyscore(self.haveMeshedPrex+self.mesh_id, currentTime, int(time.time()*10000),True)['result']
        haveMeshed = [i[0] for i in haveMeshedWithScore]
        logging.debug('['+self.mesh_id+'] meshed user:'+str(haveMeshed))
        #根据时间戳信息，返回超过10s已匹配的用户
        last10shaveMeshed = [i[0]  for i in haveMeshedWithScore if i[1]>currentTime + 290*1000]
        logging.debug('['+self.mesh_id+'] 10s or more meshed user:'+str(last10shaveMeshed))
        return self.listDifference(user, haveMeshed), self.listDifference(user, last10shaveMeshed)

    def filterFriend(self, user):
        """
        过滤用户的好友
        """
        friend = self.redis.zset_zrange(self.userFriendPrex + self.mesh_id, 0, -1)['result']
        return self.listDifference(user, friend)

    def getAccompanyChatUser(self):
        """
        获取陪聊用户列表。
        """
        result = []
        dic = self.redis.set_smembers(self.accompanyChatKey)
        #从redis取数据
        result = dic['result']
        return result

    def getMostBeauty(self, user, num):
        """
        获取用户热情度,没有特征向量的用户评分为3.5左右，取分差接近的用户进行推荐。
        """
        recResult = []
        tempRecResult = []
        #获取用户自身的评分
        selfResult = self.redis.hset_hget(self.userPopular, self.mesh_id)['result']
        if selfResult:
            selfResult = float(selfResult)
        else:
            selfResult = 3.3 + random.random()*0.4

        for i in user:
            result = self.redis.hset_hget(self.userPopular, i)['result']
            if result:
                tempRecResult.append([i, abs(float(result) - selfResult)])
            else:
                tempRecResult.append([i, abs(3.3 + random.random()*0.4 - selfResult)])

        tempRecResult.sort(key=lambda x:x[1])
        logging.debug('['+self.mesh_id+'] user score:'+str(tempRecResult))
        recResult = [tempRecResult[i][0] for i in range(len(tempRecResult))]

        return recResult[:num]


    def getNormalRec(self, user, num):
        """
        获取和当前用户匹配度最高的候选用户
        """
        #使用用户特征向量来计算匹配度
        recResult = []
        unFactorUser = []
        tempRecResult = []
        myselfFactor = self.redis.hset_hget(self.userLeft_factor, self.mesh_id)['result']
        if myselfFactor:
            myselfFactorArray = np.fromstring(myselfFactor, dtype=float)
            for i in user:
                tempFactor = self.redis.hset_hget(self.userRight_factor, i)['result']
                if tempFactor:
                    tempFactorArray = np.fromstring(tempFactor, dtype=float)
                    tempRecResult.append([i, myselfFactorArray.dot(tempFactorArray)])
                else:
                    unFactorUser.append(i)
            tempRecResult.sort(key=lambda x:-x[1])
            recResult = [tempRecResult[i][0] for i in range(len(tempRecResult))]
            if len(recResult) >= num:
                return recResult[:num]
            elif unFactorUser:
                print "getNormalRec"
                mostBeautyUser = self.getMostBeauty(unFactorUser, num-len(recResult))
                recResult.extend(mostBeautyUser)

        else:
            # 用户没有特征向量
            unFactorUser = user
            recResult = self.getMostBeauty(unFactorUser, num)
        return recResult

    def filterBlock(self, l):
        """
        过滤用户举报和屏蔽的用户
        """
        blockUser = self.redis.zset_zrange(self.blockUserPrex+self.mesh_id, 0, -1)['result']
        logging.debug('['+self.mesh_id+'] block user:'+str(blockUser))
        reportUser = self.redis.zset_zrange(self.reportUserPrex+self.mesh_id, 0, -1)['result']
        logging.debug('['+self.mesh_id+'] report user:'+str(reportUser))
        temp1 = self.listDifference(l, blockUser)
        temp2 = self.listDifference(temp1, reportUser)

        return temp2

    def getGenderMeshGender(self, id):
        """
        获取用户的性别，和期望匹配的性别
        """
        userDes = self.redis.hset_hget(self.userDetail, id)['result']
        if userDes is None:
            logging.debug('mesh_id:['+id+'] user not in redis _tt_User.')
            return 2, 99 #表示无此用户
        userDesDict = json.loads(userDes)
        gender = userDesDict['gender'] if userDesDict.has_key('gender') else 1
        meshgender = userDesDict['matchGender'] if userDesDict.has_key('matchGender') else 100 #如果没有确定的meshgender信息，则取100
        return gender, meshgender

    def getUserDetail(self, id):
        """
        获取用户的性别，期望匹配的性别，年龄，设备平台，语言, 匹配次数
        """
        ud = {}
        userDes = self.redis.hset_hget(self.userDetail, id)['result']
        if userDes is None:
            logging.debug('mesh_id:['+id+'] user not in redis _tt_User.')
            #使用默认值
            ud['gender'] = 1
            ud['meshgender'] = 100
            ud['age'] = 30
            ud['pf'] = 'IOS'
            ud['lang'] = 'en'
            return ud #表示无此用户
        userDesDict = json.loads(userDes)
        ud['gender'] = userDesDict['gender'] if userDesDict.has_key('gender') else 1
        ud['meshgender'] = userDesDict['matchGender'] if userDesDict.has_key('matchGender') else 100 #如果没有确定的meshgender信息，则取100
        ud['age'] = userDesDict['age'] if userDesDict.has_key('age') else 30 
        ud['pf'] = userDesDict['pf'] if userDesDict.has_key('pf') else 'IOS' 
        ud['lang'] = userDesDict['lang'][0:2] if userDesDict.has_key('lang') else 'en'
        redisKey = self.newPrex + str(id)
        dic = self.redis.get_data(redisKey)
        
        if dic['result'] is None:
            ud['meshcnt'] = 0
        else:
            ud['meshcnt'] = int(dic['result'])
        return ud

    def refreshGender(self, d):
        """
        为在线用户刷新性别匹配取向
        """
        for i in d:
            gender = d[i]['gender']
            meshgender = d[i]['meshgender']
            logging.debug('['+self.mesh_id+'] onlineUser refreshGender: '+str(i) + ' gender:' + str(gender) + ' meshgender:' + str(meshgender))
            if gender == 0: #女
                if meshgender == 101: #女
                    self.redis.zset_zadd(self.userGenderFemaleFemale, i, 0)
                    self.redis.zset_zrem(self.userGenderMaleFemale, i)
                    self.redis.zset_zrem(self.userGenderFemaleMale, i)
                    self.redis.zset_zrem(self.userGenderMaleMale ,i)
                elif meshgender == 102: #男
                    self.redis.zset_zadd(self.userGenderFemaleMale, i, 0)
                    self.redis.zset_zrem(self.userGenderMaleFemale, i)
                    self.redis.zset_zrem(self.userGenderMaleMale ,i)
                    self.redis.zset_zrem(self.userGenderFemaleFemale, i)
                else:
                    self.redis.zset_zadd(self.userGenderFemaleFemale, i, 0)
                    self.redis.zset_zadd(self.userGenderFemaleMale, i, 0)
                    self.redis.zset_zrem(self.userGenderMaleFemale, i)
                    self.redis.zset_zrem(self.userGenderMaleMale ,i)
    
            elif gender == 1: #男
                if meshgender == 101: #女
                    self.redis.zset_zadd(self.userGenderMaleFemale, i, 0)
                    self.redis.zset_zrem(self.userGenderFemaleMale, i)
                    self.redis.zset_zrem(self.userGenderMaleMale ,i)
                    self.redis.zset_zrem(self.userGenderFemaleFemale, i)
                elif meshgender == 102: #男
                    self.redis.zset_zadd(self.userGenderMaleMale, i, 0)
                    self.redis.zset_zrem(self.userGenderFemaleMale, i)
                    self.redis.zset_zrem(self.userGenderFemaleFemale, i)
                    self.redis.zset_zrem(self.userGenderMaleFemale, i)
                else:
                    self.redis.zset_zadd(self.userGenderMaleFemale, i, 0)
                    self.redis.zset_zadd(self.userGenderMaleMale, i, 0)
                    self.redis.zset_zrem(self.userGenderFemaleMale, i)
                    self.redis.zset_zrem(self.userGenderFemaleFemale, i)

    def userListSplitDetail(self, l, ld, selfUd):
        """
        """
        temp = l
        result = []

        sameLangSamePf = []
        sameLangDiffPf = []
        diffLangSamePf = []
        other = []

        if selfUd['age']<=18:
            ageRate = 18
            for u in temp:
                if selfUd['lang'] == ld[u]['lang'] and selfUd['pf'].upper() == ld[u]['pf'].upper() and ld[u]['age']<=ageRate:
                    sameLangSamePf.append(u)
                elif selfUd['lang'] == ld[u]['lang'] and ld[u]['age']<=ageRate:
                    sameLangDiffPf.append(u)

            result.append(sameLangSamePf)
            result.append(sameLangDiffPf)
            logging.debug('['+self.mesh_id+'] split user same lang#pf#age:'+str(sameLangSamePf))
            logging.debug('['+self.mesh_id+'] split user same lang#age:'+str(sameLangDiffPf))

            sameLangSamePf = []
            sameLangDiffPf = []

        tmp1 = self.listDifference(temp, sameLangSamePf)
        temp = self.listDifference(tmp1, sameLangDiffPf)

        for u in temp:
            if selfUd['lang'] == ld[u]['lang'] and selfUd['pf'].upper() == ld[u]['pf'].upper():
                sameLangSamePf.append(u)
            elif selfUd['lang'] == ld[u]['lang']:
                sameLangDiffPf.append(u)
            elif selfUd['pf'].upper() == ld[u]['pf'].upper():
                diffLangSamePf.append(u)
            else:
                other.append(u)

        result.append(sameLangSamePf)
        result.append(sameLangDiffPf)
        result.append(diffLangSamePf)
        result.append(other)

        logging.debug('['+self.mesh_id+'] split user same lang#pf:'+str(sameLangSamePf))
        logging.debug('['+self.mesh_id+'] split user same lang:'+str(sameLangDiffPf))
        logging.debug('['+self.mesh_id+'] split user same pf:'+str(diffLangSamePf))
        logging.debug('['+self.mesh_id+'] split user same none:'+str(other))

        return result

    def userListSplit(self, l, ld, new=False):
        """
        根据候选用户，根据规则拆分用户
        """
        result = []
        selfUd = self.getUserDetail(self.mesh_id)
        if new and selfUd['gender'] == '0':
            logging.debug('['+self.mesh_id+'] newUserSplit')
            diffGenderOld = []
            diffGenderNew = []
            sameGenderOld = []
            sameGenderNew = []
            #如果用户没有确定匹配的用户性别，才会根据性别拆分。
            if selfUd['meshgender'] == 100:
                sameGenderNew = [u for u in l if selfUd['gender'] == ld[u]['gender'] and ld[u]['meshcnt']<50]
                sameGenderOld = [u for u in l if selfUd['gender'] == ld[u]['gender'] and ld[u]['meshcnt']>=50]
                diffGenderNew = [u for u in l if selfUd['gender'] <> ld[u]['gender'] and ld[u]['meshcnt']<50]
                diffGenderOld = [u for u in l if selfUd['gender'] <> ld[u]['gender'] and ld[u]['meshcnt']>=50]
            else:
                diffGenderNew = [u for u in l if ld[u]['meshcnt']<50]
                diffGenderOld = [u for u in l if ld[u]['meshcnt']>=50]
    
            r1 = self.userListSplitDetail(diffGenderOld, ld, selfUd)
            r2 = self.userListSplitDetail(diffGenderNew, ld, selfUd)
            r3 = self.userListSplitDetail(sameGenderOld, ld, selfUd)
            r4 = self.userListSplitDetail(sameGenderNew, ld, selfUd)
    
            result.extend(r1)
            result.extend(r2)
            result.extend(r3)
            result.extend(r4)
        else:
            sameGender = []
            diffGender = []
            #如果用户没有确定匹配的用户性别，才会根据性别拆分。
            if selfUd['meshgender'] == 100:
                for u in l:
                    if selfUd['gender'] == ld[u]['gender']:
                        sameGender.append(u)
                    else:
                        diffGender.append(u)
            else:
                diffGender = l

            p = self.userListSplitDetail(diffGender, ld, selfUd)
            q = self.userListSplitDetail(sameGender, ld, selfUd)
            result.extend(p)
            result.extend(q)
        return result
        

    def getMiddleCandidate(self, key, currentTime):
        """
        获取可以匹配的候选集
        """
        onlineUser = self.redis.zset_zrangebyscore(key, currentTime-4000, currentTime)['result']
        onlineExcludeSelf = self.listDifference(onlineUser, [self.mesh_id]) #把自己从候选集排除
        matchingUser = self.redis.zset_zrangebyscore(self.matchingUserKey, currentTime-10000, currentTime)['result']
        toBeMatchedTemp = self.listDifference(onlineExcludeSelf, matchingUser) #排除已在匹配中的用户 
        matchFilter = self.filterBlock(toBeMatchedTemp)

        result = matchFilter

        logging.debug('['+self.mesh_id+'] onlineUser:'+str(onlineUser))
        logging.debug('['+self.mesh_id+'] onlineUser exclude self:'+str(onlineExcludeSelf))
        logging.debug('['+self.mesh_id+'] now matching user:'+str(matchingUser))
        logging.debug('['+self.mesh_id+'] onlineUser exclude matching user:'+str(toBeMatchedTemp))
        logging.debug('['+self.mesh_id+'] filter block user:'+str(matchFilter))

        if self.tag is None or self.tag == '':
            friendFilter = self.filterFriend(matchFilter)
            logging.debug('['+self.mesh_id+'] filter friend user:'+str(friendFilter))
            result = friendFilter

        #获取候选用户的属性
        userD = {}
        for u in result:
            ub = self.getUserDetail(u)
            userD[u]=ub
        logging.debug('['+self.mesh_id+'] get complete user detail!')
        #self.refreshGender(userD) #为在线用户刷新匹配性别匹配取向
        
        return result, userD

    def getToBeMatchedUser(self, userD, currentTime, gender, meshgender):
        """
        获取在线可以匹配的用户：获取用户对匹配性别的偏好，一个用户有gender和meshgender属性，必须条件符合的才能匹配
        在线的用户排除自己，排除已经在匹配中的用户
        """
        
        matchGender = [] #保存符合性别匹配期望的用户列表

        if gender == 1 and meshgender == 101: #男 女 匹配（女，男）或者（女，任意）
            matchGender = [u for u in userD if userD[u]['gender'] == 0 and userD[u]['meshgender'] <> 101]
        elif gender == 0 and meshgender == 101: # 女 女
            matchGender = [u for u in userD if userD[u]['gender'] == 0 and userD[u]['meshgender'] <> 102]
        elif gender == 1 and meshgender == 102:
            matchGender = [u for u in userD if userD[u]['gender'] == 1 and userD[u]['meshgender'] <> 101]
        elif gender == 0 and meshgender == 102:
            matchGender = [u for u in userD if userD[u]['gender'] == 1 and userD[u]['meshgender'] <> 102]
        elif gender == 1 and meshgender == 100: #男 任意 匹配 （女，男），（女，任意），（男，男），（男，任意）
            matchGender = [u for u in userD if userD[u]['meshgender'] <> 101]
        elif gender == 0 and meshgender == 100:
            matchGender = [u for u in userD if userD[u]['meshgender'] <> 102]

        """
        if gender == 0: #女
            if meshgender == 101: #女
                matchGender = self.redis.zset_zrange(self.userGenderFemaleFemale, 0, -1)['result']
                self.redis.zset_zadd(self.userGenderFemaleFemale, self.mesh_id, 0)
                self.redis.zset_zrem(self.userGenderMaleFemale, self.mesh_id)
                self.redis.zset_zrem(self.userGenderFemaleMale, self.mesh_id)
                self.redis.zset_zrem(self.userGenderMaleMale ,self.mesh_id)
            elif meshgender == 102: #男
                matchGender = self.redis.zset_zrange(self.userGenderMaleFemale, 0, -1)['result']
                self.redis.zset_zadd(self.userGenderFemaleMale, self.mesh_id, 0)
                self.redis.zset_zrem(self.userGenderMaleFemale, self.mesh_id)
                self.redis.zset_zrem(self.userGenderMaleMale ,self.mesh_id)
                self.redis.zset_zrem(self.userGenderFemaleFemale, self.mesh_id)
            else:
                matchGender = self.redis.zset_zrange(self.userGenderMaleFemale, 0, -1)['result']
                matchGender.extend(self.redis.zset_zrange(self.userGenderFemaleFemale, 0, -1)['result'])
                self.redis.zset_zadd(self.userGenderFemaleFemale, self.mesh_id, 0)
                self.redis.zset_zadd(self.userGenderFemaleMale, self.mesh_id, 0)
                self.redis.zset_zrem(self.userGenderMaleFemale, self.mesh_id)
                self.redis.zset_zrem(self.userGenderMaleMale ,self.mesh_id)
        elif gender == 1: #男
            if meshgender == 101: #女
                matchGender = self.redis.zset_zrange(self.userGenderFemaleMale, 0, -1)['result']
                self.redis.zset_zadd(self.userGenderMaleFemale, self.mesh_id, 0)
                self.redis.zset_zrem(self.userGenderFemaleMale, self.mesh_id)
                self.redis.zset_zrem(self.userGenderMaleMale ,self.mesh_id)
                self.redis.zset_zrem(self.userGenderFemaleFemale, self.mesh_id)
            elif meshgender == 102: #男
                matchGender = self.redis.zset_zrange(self.userGenderMaleMale, 0, -1)['result']
                self.redis.zset_zadd(self.userGenderMaleMale, self.mesh_id, 0)
                self.redis.zset_zrem(self.userGenderFemaleMale, self.mesh_id)
                self.redis.zset_zrem(self.userGenderFemaleFemale, self.mesh_id)
                self.redis.zset_zrem(self.userGenderMaleFemale, self.mesh_id)
            else:
                matchGender = self.redis.zset_zrange(self.userGenderFemaleMale, 0, -1)['result']
                matchGender.extend(self.redis.zset_zrange(self.userGenderMaleMale, 0, -1)['result'])
                self.redis.zset_zadd(self.userGenderMaleFemale, self.mesh_id, 0)
                self.redis.zset_zadd(self.userGenderMaleMale, self.mesh_id, 0)
                self.redis.zset_zrem(self.userGenderFemaleMale, self.mesh_id)
                self.redis.zset_zrem(self.userGenderFemaleFemale, self.mesh_id)
        """

        toBeMatched = matchGender
        logging.debug('['+self.mesh_id+'] can match user:'+str(toBeMatched))
        historyFilter, historyFilterLast10s  = self.filterHaveMeshed(toBeMatched, currentTime)
        
        logging.debug('['+self.mesh_id+'] filter have mesh user:'+str(historyFilter))
        logging.debug('['+self.mesh_id+'] filter have mesh user 10s or more:'+str(historyFilterLast10s))

        return historyFilter, historyFilterLast10s

    def accompanyRec(self, toBeMatchedAccompany, accompanyUserRec, currentTime):
        """
        根据陪聊用户的匹配时间,排序陪聊用户
        """
        toBeMatchedSort = []
        for i in toBeMatchedAccompany:
            if accompanyUserRec:
                for j in range(len(accompanyUserRec)):
                    if accompanyUserRec[j][0] == i:
                        toBeMatchedSort.append((i, float(accompanyUserRec[j][1])))
                        break
                    if j == len(accompanyUserRec)-1:
                        toBeMatchedSort.append((i, 0))
            else:
                toBeMatchedSort.append((i, 0))
        toBeMatchedSort.sort(key=lambda x:x[1])
        toBeMatch = [i[0] for i in toBeMatchedSort]

        #记录匹配的陪聊用户的结果到redis
        if toBeMatch:
            self.redis.zset_zadd(self.accompanyUserRecSort, toBeMatch[0], currentTime) #修改陪聊用户的被推荐时间
        return toBeMatch

    def makeNewIDRecDetail(self, accompanyChatUser, toBeMatched, currentTime, userD):
        """
        完成新用户推荐的工作。
        """
        recResult = []
        accompanyUserRec = self.redis.zset_zrangebyscore(self.accompanyUserRecSort, currentTime-1000*3600*24, currentTime, True)['result'] #拿出陪聊用户的历史匹配次序
        toBeMatchedAccompany = self.listIntersection(accompanyChatUser, toBeMatched) #在线的陪聊用户

        if toBeMatchedAccompany:
            ulAll = self.userListSplit(toBeMatchedAccompany, userD, True)
            for ul in ulAll:
                if ul:
                    recResult.extend(self.accompanyRec(ul, accompanyUserRec, currentTime))

            if len(recResult) >= self.recLen:
                return recResult[:self.recLen]

        onlineMatchUser = self.listDifference(toBeMatched, accompanyChatUser) #在线的普通用户
        if onlineMatchUser:
            ulAll = self.userListSplit(onlineMatchUser, userD, True)
            for ul in ulAll:
                if ul:
                    recResult.extend(self.getMostBeauty(ul, self.recLen))
        return recResult

    def makeNewIDRec(self):
        """
        为新用户生成推荐结果，优先推荐陪聊用户，再推荐普通用户
        """
        recResult = []
        currentTime = int(time.time()*1000)

        accompanyChatUser = self.getAccompanyChatUser()

        gender, meshgender = self.getGenderMeshGender(self.mesh_id)
        print gender, meshgender
        if gender == 2:
            return []
        elif meshgender == 100:
            middleCandidate, userD = self.getMiddleCandidate(self.onlineUserKey ,currentTime)
            if middleCandidate:
                toBeMatched, toBeMatchedLast10s = self.getToBeMatchedUser(userD ,currentTime, gender, meshgender)
                if toBeMatched:
                    recResult = self.makeNewIDRecDetail(accompanyChatUser, toBeMatched, currentTime, userD)
                elif toBeMatchedLast10s:
                    recResult = self.makeNewIDRecDetail(accompanyChatUser, toBeMatchedLast10s, currentTime, userD)

        else:
            middleCandidate, userD = self.getMiddleCandidate(self.onlineUserKey ,currentTime)
            if middleCandidate:
                toBeMatched, toBeMatchedLast10s = self.getToBeMatchedUser(userD ,currentTime, gender, meshgender)
                if toBeMatched:
                    recResult = self.makeNewIDRecDetail(accompanyChatUser, toBeMatched, currentTime, userD)
                elif toBeMatchedLast10s:
                    recResult = self.makeNewIDRecDetail(accompanyChatUser, toBeMatchedLast10s, currentTime, userD)

        return recResult[:self.recLen]

    def makeOldIDRecDetail(self, accompanyChatUser, toBeMatched, currentTime, userD):
        """
        按照顺序给老用户推荐，先推荐普通用户，再推荐陪聊用户
        """
        recResult = []
        matcheUserExcludeAccompany = self.listDifference(toBeMatched, accompanyChatUser) # 排除陪聊用户的待匹配用户
        if matcheUserExcludeAccompany:
            #普通用户推荐
            ulAll = self.userListSplit(matcheUserExcludeAccompany, userD)
            for ul in ulAll:
                if ul:
                    recResult.extend(self.getMostBeauty(ul, self.recLen))

            if len(recResult) < self.recLen:
                accompanyUserRec = self.redis.zset_zrangebyscore(self.accompanyUserRecSort, currentTime-1000*3600*24, currentTime, True)['result'] #拿出陪聊用户的历史匹配次序
                toBeMatchedAccompany = self.listIntersection(accompanyChatUser, toBeMatched)
                if toBeMatchedAccompany:
                    ulAll = self.userListSplit(toBeMatchedAccompany, userD)
                    for ul in ulAll:
                        if ul:
                            recResult.extend(self.accompanyRec(ul, accompanyUserRec, currentTime))
            else:
                #推荐列表长度足够
                pass
        else:
            #没有普通用户，推荐陪聊用户
            accompanyUserRec = self.redis.zset_zrangebyscore(self.accompanyUserRecSort, currentTime-1000*3600*24, currentTime, True)['result'] #拿出陪聊用户的历史匹配次序
            toBeMatchedAccompany = self.listIntersection(accompanyChatUser, toBeMatched)
            if toBeMatchedAccompany:
                ulAll = self.userListSplit(toBeMatchedAccompany, userD)
                for ul in ulAll:
                    if ul:
                        recResult.extend(self.accompanyRec(ul, accompanyUserRec, currentTime))

        return recResult

    def makeOldIDRec(self):
        """
        给老用户进行推荐
        """
        recResult = []
        accompanyChatUser = self.getAccompanyChatUser()
        if self.mesh_id in accompanyChatUser:
            logging.debug('mesh_id:['+self.mesh_id+'] It`s an accompany user.')
            return recResult

        currentTime = int(time.time()*1000)
        gender, meshgender = self.getGenderMeshGender(self.mesh_id)
        if gender == 2:
            return []
        elif meshgender == 100:
            middleCandidate, userD = self.getMiddleCandidate(self.onlineUserKey ,currentTime)
            if middleCandidate:
                toBeMatched, toBeMatchedLast10s = self.getToBeMatchedUser(userD ,currentTime, gender, meshgender)
                if toBeMatched:
                    recResult = self.makeOldIDRecDetail(accompanyChatUser, toBeMatched, currentTime, userD)
                elif toBeMatchedLast10s:
                    recResult = self.makeOldIDRecDetail(accompanyChatUser, toBeMatchedLast10s, currentTime, userD)

        else:
            middleCandidate, userD = self.getMiddleCandidate(self.onlineUserKey ,currentTime)
            if middleCandidate:
                toBeMatched, toBeMatchedLast10s = self.getToBeMatchedUser(userD ,currentTime, gender, meshgender)
                if toBeMatched:
                    recResult = self.makeOldIDRecDetail(accompanyChatUser, toBeMatched, currentTime, userD)
                elif toBeMatchedLast10s:
                    recResult = self.makeOldIDRecDetail(accompanyChatUser, toBeMatchedLast10s, currentTime, userD)

        return recResult[:self.recLen]

    def rec(self):
        """
        推荐入口，新老用户区别对待
        """
        logging.debug('mesh_id:['+self.mesh_id+'] Begin Connect')
        if self.ifDoubtful():
            logging.debug('['+self.mesh_id+'] match result user: []')
            return []
        elif self.ifNewID():
            print "newnewnewnew"
            recResult = self.makeNewIDRec()
        elif self.luckyTag():
            recResult = self.makeNewIDRec()
        else:
            recResult = self.makeOldIDRec()
        self.close()
        logging.debug('['+self.mesh_id+'] match result user:'+str(recResult))
        return recResult

    def close(self):
        """
        释放redis连接
        """
        self.redis.release()

if __name__ == '__main__':
    mesh_id = '95580139'
    tag = ''
    ver = 'ver'

    import datetime
    print datetime.datetime.now()
    rec = Recommend(mesh_id, tag, ver)
    result = rec.rec()
    print result
    print datetime.datetime.now()

