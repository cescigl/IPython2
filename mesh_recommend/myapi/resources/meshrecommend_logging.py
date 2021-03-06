# -*- coding: utf-8 -*-
from flask import Flask
from flask_restful import Resource, Api, reqparse, fields, marshal_with
import time
import random 
import json
import numpy as np
import sys
sys.path.append("..")
from common.redispool import RedisCache
#from common.ddbpool import DDB
import logging 
from logging.handlers import TimedRotatingFileHandler   
import os  
  

class MeshRecommend(Resource):
    def get(self, arg):
        # arg =  "mesh_id=123&tag=456"
        print arg
        mesh_id = arg.split("&")[0].split('=')[1]
        tag = arg.split("&")[1].split('=')[1]
        ver = arg.split("&")[2].split('=')[1]
        print mesh_id   
        print tag
        print ver

        rec = Recommend(mesh_id, tag, ver)
        result = rec.rec()
        return {'result':'#'.join(result)} # 用#拼接的字符串。

class Recommend(object):
    """ make Recommend result """
    def __init__(self, mesh_id, tag='', ver=''):
        super(Recommend, self).__init__()

        self.log_filename="mesh_matching_logging.txt"
        self.datefmt = '%Y-%m-%d %H:%M:%S'
        self.log_format=' [%(asctime)s]   %(message)s' 
        self.formatter = logging.Formatter(self.log_format, self.datefmt)

        self.log_file_handler = TimedRotatingFileHandler(filename=self.log_filename, when="M", interval=1, backupCount=20)
        self.log_file_handler.setFormatter(self.formatter)  
        #将日志文件格式化  
        #logging.basicConfig(datafmt='%Y-%m-%d %H:%M:%S %p',level=logging.DEBUG) 
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        if not len(self.log.handlers):
            self.log.addHandler(self.log_file_handler) 


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
        self.userFriendPrex = 'rec_user_friend_' #记录用户的好友列表 （set）

        self.userLeft_factor = "rec_userLeft_factor" #记录用户特征向量
        self.userRight_factor = "rec_userRight_factor" 
        self.userPopular = 'rec_popular' # 记录用户的流行度 hset
        

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
            if num <= 2:
                ifNew = True
                return ifNew
            elif num <= 5:
                ifNew = (True if random.random()<0.3 else False)
                return ifNew

        # 老用户20%的概率以老用户处理
        ifNew = (True if random.random()<0.2 else False)
        return ifNew
    
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
        haveMeshed = self.redis.zset_zrangebyscore(self.haveMeshedPrex+self.mesh_id, currentTime, int(time.time()*10000))['result']
        self.log.info('['+self.mesh_id+'] meshed user:'+str(haveMeshed))
        
        return self.listDifference(user, haveMeshed)

    def filterFriend(self, user):
        """
        过滤用户的好友
        """
        friend = self.redis.set_smembers(self.userFriendPrex + self.mesh_id)['result']
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
        获取用户热情度最高的用户,没有特征向量的用户随机排序
        """
        recResult = []
        unBeautyUser = []
        tempRecResult = []
        for i in user:
            result = self.redis.hset_hget(self.userPopular, i)['result']
            if result:
                tempRecResult.append([i, float(result)])
            else:
                unBeautyUser.append(i)

        tempRecResult.sort(key=lambda x:-x[1])
        self.log.info('['+self.mesh_id+'] user score:'+str(tempRecResult))
        recResult = [tempRecResult[i][0] for i in range(len(tempRecResult))]

        if len(tempRecResult) >= num:
            return recResult[:num]
        else:
            random.shuffle(unBeautyUser)
            recResult.extend(unBeautyUser)

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
        self.log.info('['+self.mesh_id+'] block user:'+str(blockUser))
        reportUser = self.redis.zset_zrange(self.reportUserPrex+self.mesh_id, 0, -1)['result']
        self.log.info('['+self.mesh_id+'] report user:'+str(reportUser))
        temp1 = self.listDifference(l, blockUser)
        temp2 = self.listDifference(temp1, reportUser)

        return temp2

    def getGenderMeshGender(self, id):
        """
        获取用户的性别，和期望匹配的性别
        """
        userDes = self.redis.hset_hget(self.userDetail, id)['result']
        if userDes is None:
            self.log.info('mesh_id:['+id+'] user not in redis _tt_User.')
            return 2, 99 #表示无此用户
        userDesDict = json.loads(userDes)
        gender = userDesDict['gender']
        meshgender = userDesDict['matchGender'] if userDesDict.has_key('matchGender') else 100 #如果没有确定的meshgender信息，则取100
        return gender, meshgender

    def refreshGender(self, l):
        """
        为在线用户刷新性别匹配取向
        """
        for i in l:
            gender, meshgender = self.getGenderMeshGender(i)
            self.log.info('['+self.mesh_id+'] onlineUser refreshGender: '+str(i) + ' gender:' + str(gender) + ' meshgender:' + str(meshgender)) 
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


    def getToBeMatchedUser(self, key, currentTime, gender, meshgender, status=1):
        """
        获取在线可以匹配的用户：获取用户对匹配性别的偏好，一个用户有gender和meshgender属性，必须条件符合的才能匹配
        在线的用户排除自己，排除已经在匹配中的用户
        """

        onlineUser = self.redis.zset_zrangebyscore(key, currentTime-4000, currentTime)['result']
        onlineExcludeSelf = self.listDifference(onlineUser, [self.mesh_id]) #把自己从候选集排除
        matchingUser = self.redis.zset_zrangebyscore(self.matchingUserKey, currentTime-10000, currentTime)['result']
        toBeMatchedTemp = self.listDifference(onlineExcludeSelf, matchingUser) #排除已在匹配中的用户 

        self.refreshGender(toBeMatchedTemp) #为在线用户刷新匹配性别匹配取向

        matchGender = [] #保存符合性别匹配期望的用户列表
        
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
                if status == 1: #异性
                    matchGender = self.redis.zset_zrange(self.userGenderMaleFemale, 0, -1)['result']
                    self.redis.zset_zadd(self.userGenderFemaleFemale, self.mesh_id, 0)
                    self.redis.zset_zadd(self.userGenderFemaleMale, self.mesh_id, 0)
                    self.redis.zset_zrem(self.userGenderMaleFemale, self.mesh_id)
                    self.redis.zset_zrem(self.userGenderMaleMale ,self.mesh_id)
                elif status == 2: #同性  
                    matchGender = self.redis.zset_zrange(self.userGenderFemaleFemale, 0, -1)['result']
                else:
                    matchGender = self.redis.zset_zrange(self.userGenderMaleFemale, 0, -1)['result']
                    matchGender.extend(self.redis.zset_zrange(self.userGenderFemaleFemale, 0, -1)['result'])

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
                if status == 1: #异性
                    matchGender = self.redis.zset_zrange(self.userGenderFemaleMale, 0, -1)['result']
                    self.redis.zset_zadd(self.userGenderMaleFemale, self.mesh_id, 0)
                    self.redis.zset_zadd(self.userGenderMaleMale, self.mesh_id, 0)
                    self.redis.zset_zrem(self.userGenderFemaleMale, self.mesh_id)
                    self.redis.zset_zrem(self.userGenderFemaleFemale, self.mesh_id)
                elif status == 2: #同性
                    matchGender = self.redis.zset_zrange(self.userGenderMaleMale, 0, -1)['result']
                else:
                    matchGender = self.redis.zset_zrange(self.userGenderFemaleMale, 0, -1)['result']
                    matchGender.extend(self.redis.zset_zrange(self.userGenderMaleMale, 0, -1)['result'])
    

        toBeMatched = self.listIntersection(toBeMatchedTemp, matchGender)
        matchFilter = self.filterBlock(toBeMatched)
        historyFilter = self.filterHaveMeshed(matchFilter, currentTime)
        if status == 3:
            historyFilter = self.filterHaveMeshed(matchFilter, currentTime + 240*1000)
        self.log.info('['+self.mesh_id+'] onlineUser:'+str(onlineUser))
        self.log.info('['+self.mesh_id+'] onlineUser exclude self:'+str(onlineExcludeSelf))
        self.log.info('['+self.mesh_id+'] now matching user:'+str(matchingUser))
        self.log.info('['+self.mesh_id+'] onlineUser exclude matching user:'+str(toBeMatchedTemp))

        self.log.info('['+self.mesh_id+'] can match user:'+str(toBeMatched))
        self.log.info('['+self.mesh_id+'] filter block user:'+str(matchFilter))
        self.log.info('['+self.mesh_id+'] filter have mesh user:'+str(historyFilter))

        if self.tag is None or self.tag == '':
            friendFilter = self.filterFriend(historyFilter)
            self.log.info('['+self.mesh_id+'] filter friend user:'+str(friendFilter))
            return friendFilter

        return historyFilter

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


    def makeNewIDRecDetail(self, accompanyChatUser, toBeMatched, currentTime):
        """
        完成新用户推荐的工作。
        """
        recResult = []
        accompanyUserRec = self.redis.zset_zrangebyscore(self.accompanyUserRecSort, currentTime-1000*3600*24, currentTime, True)['result'] #拿出陪聊用户的历史匹配次序
        toBeMatchedAccompany = self.listIntersection(accompanyChatUser, toBeMatched) #在线的陪聊用户

        if toBeMatchedAccompany:
            toBeMatch =  self.accompanyRec(toBeMatchedAccompany, accompanyUserRec, currentTime)
            recResult.extend(toBeMatch)

            if len(recResult) >= self.recLen:
                return recResult[:self.recLen]

        onlineMatchUser = self.listDifference(toBeMatched, accompanyChatUser) #在线的普通用户
        mostBeauty = self.getMostBeauty(onlineMatchUser, self.recLen)
        recResult.extend(mostBeauty)
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
            toBeMatched = self.getToBeMatchedUser(self.onlineUserKey ,currentTime, gender, meshgender, 1)
            if toBeMatched:
                recResult = self.makeNewIDRecDetail(accompanyChatUser, toBeMatched, currentTime)

            if len(recResult) >= self.recLen:
                return recResult[:self.recLen]
            else:
                toBeMatched = self.getToBeMatchedUser(self.onlineUserKey ,currentTime, gender, meshgender, 2)
                if toBeMatched:
                    recResult.extend(self.makeNewIDRecDetail(accompanyChatUser, toBeMatched, currentTime))
                    return recResult[:self.recLen]
                elif len(recResult) == 0: #没有符合条件的用户
                    print "no fit"
                    recResult = self.getToBeMatchedUser(self.onlineUserKey ,currentTime, gender, meshgender, 3)
                    random.shuffle(recResult)

        else:
            toBeMatched = self.getToBeMatchedUser(self.onlineUserKey ,currentTime, gender, meshgender)
            if toBeMatched:
                recResult = self.makeNewIDRecDetail(accompanyChatUser, toBeMatched, currentTime)
            else:#没有符合条件的用户
                print "no fit"
                recResult = self.getToBeMatchedUser(self.onlineUserKey ,currentTime, gender, meshgender, 3)
                random.shuffle(recResult)

        return recResult[:self.recLen]

    def makeOldIDRecDetail(self, accompanyChatUser, toBeMatched, currentTime):
        """
        按照顺序给老用户推荐，先推荐普通用户，再推荐陪聊用户
        """
        recResult = []
        matcheUserExcludeAccompany = self.listDifference(toBeMatched, accompanyChatUser) # 排除陪聊用户的待匹配用户
        if matcheUserExcludeAccompany:
            #普通用户推荐
            recResult = self.getMostBeauty(matcheUserExcludeAccompany, self.recLen)

            if len(recResult) < self.recLen:
                accompanyUserRec = self.redis.zset_zrangebyscore(self.accompanyUserRecSort, currentTime-1000*3600*24, currentTime, True)['result'] #拿出陪聊用户的历史匹配次序
                toBeMatchedAccompany = self.listIntersection(accompanyChatUser, toBeMatched)
                toBeMatch = self.accompanyRec(toBeMatchedAccompany, accompanyUserRec, currentTime)
                recResult.extend(toBeMatch)
            else:
                #推荐列表长度足够
                pass
        else:
            #没有普通用户，推荐陪聊用户
            accompanyUserRec = self.redis.zset_zrangebyscore(self.accompanyUserRecSort, currentTime-1000*3600*24, currentTime, True)['result'] #拿出陪聊用户的历史匹配次序
            toBeMatchedAccompany = self.listIntersection(accompanyChatUser, toBeMatched)
            recResult = self.accompanyRec(toBeMatchedAccompany, accompanyUserRec, currentTime)

        return recResult

    def makeOldIDRec(self):
        """
        给老用户进行推荐
        """
        recResult = []
        accompanyChatUser = self.getAccompanyChatUser()
        if self.mesh_id in accompanyChatUser:
            self.log.info('mesh_id:['+self.mesh_id+'] It`s an accompany user.')
            return recResult

        currentTime = int(time.time()*1000)
        gender, meshgender = self.getGenderMeshGender(self.mesh_id)
        if gender == 2:
            return []
        elif meshgender == 100:
            toBeMatched = self.getToBeMatchedUser(self.onlineUserKey ,currentTime, gender, meshgender, 1)
            if toBeMatched:
                recResult = self.makeOldIDRecDetail(accompanyChatUser, toBeMatched, currentTime)

            if len(recResult) >= self.recLen:
                return recResult[:self.recLen]
            else:
                toBeMatched = self.getToBeMatchedUser(self.onlineUserKey ,currentTime, gender, meshgender, 2)
                if toBeMatched:
                    recResult.extend(self.makeOldIDRecDetail(accompanyChatUser, toBeMatched, currentTime))
                    return recResult[:self.recLen]
                elif len(recResult) == 0:
                    recResult = self.getToBeMatchedUser(self.onlineUserKey ,currentTime, gender, meshgender, 3)
                    random.shuffle(recResult)

        else:
            toBeMatched = self.getToBeMatchedUser(self.onlineUserKey ,currentTime, gender, meshgender)
            if toBeMatched:
                recResult = self.makeOldIDRecDetail(accompanyChatUser, toBeMatched, currentTime)
            else:
                recResult = self.getToBeMatchedUser(self.onlineUserKey ,currentTime, gender, meshgender, 3)
                random.shuffle(recResult)

        return recResult[:self.recLen]

    def rec(self):
        """
        推荐入口，新老用户区别对待
        """
        self.log.info('mesh_id:['+self.mesh_id+'] Begin Connect')
        if self.ifNewID():
            print "newnewnewnew"
            recResult = self.makeNewIDRec()
        elif self.luckyTag():
            recResult = self.makeNewIDRec()
        else:
            recResult = self.makeOldIDRec()
        self.close()
        self.log.info('['+self.mesh_id+'] match result user:'+str(recResult))
        self.log.removeHandler(self.log_file_handler)
        return recResult

    def close(self):
        """
        释放redis连接
        """
        self.redis.release()

if __name__ == '__main__':
    mesh_id = '97836494'
    mesh_id = '123456788'
    tag = ''
    ver = 'ver'

    import datetime
    print datetime.datetime.now()
    rec = Recommend(mesh_id, tag, ver)
    result = rec.rec()
    print result
    print datetime.datetime.now()

