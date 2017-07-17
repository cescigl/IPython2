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


class MeshRecommend(Resource):
    def get(self, arg):
        # arg =  "meshid=123&tag=456"
        if "&" in arg:
            mesh_id = arg.split("&")[0].split('=')[1]
            tag = arg.split("&")[1].split('=')[1]
        else:
            mesh_id = arg.split('=')[1]
            tag = None
        rec = Recommend(mesh_id, tag)
        result = rec.rec()
        return '#'.join(result) # 用#拼接的字符串。

class Recommend(object):
    """ make Recommend result """
    def __init__(self, mesh_id, tag=None):
        super(Recommend, self).__init__()
        self.mesh_id = mesh_id #mesh id
        self.tag = tag #匹配暗号

        self.redis = RedisCache()
        #self.ddb = DDB()
        self.recLen = 10 # 推荐结果的长度

        self.accompanyChatKey = '_tt_fake_user_id_list' #陪聊redis key（zset）
        self.onlineUserKey = '_tt_waiting_4_match_list'  #在线用户 redis key，有效数据时4s内活跃数据
        self.onlineUserTagKeyPrex = '_tt_waiting_4_match_list_'  #使用暗号匹配的在线用户 redis key，有效数据时4s内活跃数据
        self.matchingUserKey = '_tt_locked_user_list' #正在匹配中的用户  redis key，有效数据时10s内活跃数据
        self.userDetail = '_tt_User' #保存用户的所有属性， hset
        self.blockUserPrex = '_tt_user_black_list_' #记录被用户拉黑的用户
        self.reportUserPrex = '_tt_user_report_list_' # 记录被用户举报的用户


        #gender 0-女性 1-男性 meshgender 101-女性 102-男性
        self.userGenderMaleMale = 'rec_user_gender_male_male' # 保存男性用户愿意匹配男性用户的列表
        self.userGenderMaleFemale = 'rec_user_gender_male_female' # 保存男性用户愿意匹配女性用户的列表
        self.userGenderFemaleMale = 'rec_user_gender_female_male' # 保存女性用户愿意匹配男性用户的列表
        self.userGenderFemaleFemale = 'rec_user_gender_female_female' # 保存女性用户愿意匹配女性用户的列表

        self.newPrex = 'rec_hava_matched_' #记录匹配过的用户，匹配过即变成老用户
        self.accompanyUserRecSort = 'rec_accompany_user_sort' #记录陪聊用户的最新一次被推荐时间。

        self.userLeft_factor = "rec_userLeft_factor" #记录用户特征向量
        self.userRight_factor = "rec_userRight_factor"
        self.userPopular = 'rec_popular' # 记录用户的流行度 hset

    def ifNewID(self): #已测试
        """ 
        通过用户信息表判断是否是新用户，通过过去2小时的用户匹配请求判断是否是新用户 
        """
        ifNew = False
        #通过查询redis判断是否在过去2小时有过匹配 , rec_hava_matched_mesh_id
        redisKey = self.newPrex + self.mesh_id

        dic = self.redis.get_data(redisKey)
        self.redis.set_data(redisKey, 0)
        self.redis.set_expire(redisKey, 86400)

        if dic['result'] is None:
            ifNew = True
            return ifNew

        return ifNew

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

    def haveMeshed(self): #未完
        """
        过滤已经匹配过的用户
        """
        return []

    def getAccompanyChatUser(self): #已测试
        """
        获取陪聊用户列表。
        """
        result = []
        dic = self.redis.zset_zrange(self.accompanyChatKey, 0, -1)
        #从redis取数据
        result = dic['result']
        return result

    def getMostBeauty(self, user, num): #已测试
        """
        获取用户热情度最高的用户
        """
        recResult = []
        unBeautyUser = []
        tempRecResult = []
        for i in user:
            result = self.redis.hset_hget(self.userPopular, i)['result']
            if result:
                tempRecResult.append([i, result])
            else:
                unBeautyUser.append(i)

        if len(tempRecResult) >= num:
            tempRecResult.sort(key=lambda x:-x[1])
            recResult = [tempRecResult[i][0] for i in range(len(tempRecResult))]
        else:
            random.shuffle(unBeautyUser)
            recResult.extend(unBeautyUser)
        print "getMostBeauty"
        print recResult
        return recResult[:num]


    def getNormalRec(self, user, num):
        """
        获取和当前用户匹配度最高的候选用户
        """
        #使用用户特征向量来计算匹配度
        print "getNormalRec"

        recResult = []
        unFactorUser = []
        tempRecResult = []
        myselfFactor = self.redis.hset_hget(self.userLeft_factor, self.mesh_id)['result']
        print "myselfFactor"
        print myselfFactor
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
            mostBeautyUser = self.getMostBeauty(unFactorUser, num)
            recResult.extend(mostBeautyUser)
        return recResult

    def filterBlock(self, l): #已测试
        """
        过滤用户举报和屏蔽的用户
        """
        blockUser = self.redis.zset_zrange(self.blockUserPrex, 0, -1)['result']
        reportUser = self.redis.zset_zrange(self.reportUserPrex, 0, -1)['result']
        temp1 = self.listDifference(l, blockUser)
        temp2 = self.listDifference(temp1, reportUser)
        return temp1

    def getToBeMatchedUser(self, key, currentTime): #已测试
        """
        获取在线可以匹配的用户：获取用户对匹配性别的偏好，一个用户有gender和meshgender属性，必须条件符合的才能匹配
        在线的用户排除自己，排除已经在匹配中的用户
        """
        matchGender = [] #保存符合性别匹配期望的用户列表

        userDes = self.redis.hset_hget(self.userDetail, self.mesh_id)['result']

        if userDes is None:
            print "userDes is None"
            return []
        userDesDict = json.loads(userDes)
        gender = userDesDict['gender']
        meshgender = userDesDict['meshgender'] if userDesDict.has_key('meshgender') else 100 #如果没有确定的meshgender信息，则取100
        if gender == 0:
            if meshgender == 101:
                matchGender = self.redis.zset_zrange(self.userGenderFemaleFemale, 0, -1)['result']
                self.redis.zset_zadd(self.userGenderFemaleFemale, self.mesh_id, 0)
                self.redis.zset_zrem(self.userGenderMaleFemale, self.mesh_id)
                self.redis.zset_zrem(self.userGenderFemaleMale, self.mesh_id)
                self.redis.zset_zrem(self.userGenderMaleMale ,self.mesh_id)
            elif meshgender == 102:
                matchGender = self.redis.zset_zrange(self.userGenderMaleFemale, 0, -1)['result']
                self.redis.zset_zadd(self.userGenderFemaleMale, self.mesh_id, 0)
                self.redis.zset_zrem(self.userGenderMaleFemale, self.mesh_id)
                self.redis.zset_zrem(self.userGenderMaleMale ,self.mesh_id)
                self.redis.zset_zrem(self.userGenderFemaleFemale, self.mesh_id)
            else:
                matchGender = self.redis.zset_zrange(self.userGenderFemaleFemale, 0, -1)['result']
                matchGender.extend(self.redis.zset_zrange(self.userGenderMaleFemale, 0, -1)['result'])
                self.redis.zset_zadd(self.userGenderFemaleFemale, self.mesh_id, 0)
                self.redis.zset_zadd(self.userGenderFemaleMale, self.mesh_id, 0)
                self.redis.zset_zrem(self.userGenderMaleFemale, self.mesh_id)
                self.redis.zset_zrem(self.userGenderMaleMale ,self.mesh_id)

        elif gender == 1:
            if meshgender == 101:
                matchGender = self.redis.zset_zrange(self.userGenderFemaleMale, 0, -1)['result']
                self.redis.zset_zadd(self.userGenderMaleFemale, self.mesh_id, 0)
                self.redis.zset_zrem(self.userGenderFemaleMale, self.mesh_id)
                self.redis.zset_zrem(self.userGenderMaleMale ,self.mesh_id)
                self.redis.zset_zrem(self.userGenderFemaleFemale, self.mesh_id)
            elif meshgender == 102:
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
        
        onlineUser = self.redis.zset_zrange(key, 0, -1)['result']

        onlineExcludeSelf = self.listDifference(onlineUser, [self.mesh_id]) #把自己从候选集排除
        matchingUser = self.redis.zset_zrangebyscore(self.matchingUserKey, 0, currentTime)['result'] 
        toBeMatchedTemp = self.listDifference(onlineExcludeSelf, matchingUser) #排除已在匹配中的用户 
        toBeMatched = self.listIntersection(toBeMatchedTemp, matchGender)
        matchFilter = self.filterBlock(toBeMatched)
        print "matchFilter"
        print matchFilter

        return matchFilter

    def accompanyRec(self, toBeMatchedAccompany, accompanyUserRec): #已测试
        #根据陪聊用户的匹配时间,排序陪聊用户
        toBeMatchedSort = []
        for i in toBeMatchedAccompany:
            for j in range(len(accompanyUserRec)):
                if accompanyUserRec[j][0] == i:
                    toBeMatchedSort.append((i, accompanyUserRec[j][1]))
                    break
                if j == len(accompanyUserRec)-1:
                    toBeMatchedSort.append((i, 0))
        toBeMatchedSort.sort(key=lambda x:x[1])
        toBeMatch = [i[0] for i in toBeMatchedSort]

        #记录匹配的陪聊用户的结果到redis
        if toBeMatch:
            self.redis.zset_zadd(self.accompanyUserRecSort, toBeMatch[0], currentTime) #修改陪聊用户的被推荐时间
        return toBeMatch

    def makeNewIDRec(self):
        """
        为新用户生成推荐结果
        """
        recResult = []
        currentTime = int(time.time())

        accompanyChatUser = self.getAccompanyChatUser()
        toBeMatched = self.getToBeMatchedUser(self.onlineUserKey, currentTime)

        if toBeMatched:
            #有在线用户
            accompanyUserRec = self.redis.zset_zrangebyscore(self.accompanyUserRecSort, 0, currentTime, True)['result'] #拿出陪聊用户的历史匹配次序
            toBeMatchedAccompany = self.listIntersection(accompanyChatUser, toBeMatched) #在线的陪聊用户

            if toBeMatchedAccompany:
                toBeMatch =  self.accompanyRec(toBeMatchedAccompany, accompanyUserRec)
                recResult.extend(toBeMatch)

            onlineMatchUser = self.listDifference(toBeMatched, accompanyChatUser) #在线的普通用户

            mostBeauty = self.getMostBeauty(onlineMatchUser, self.recLen)
            recResult.extend(mostBeauty)
        else:
            #无在线用户
            pass

        return recResult[:self.recLen]

    def makeOldIDRecDetail(self, toBeMatched, currentTime, accompanyChatUser): #已测试
        recResult = []
        matcheUserExcludeAccompany = self.listDifference(toBeMatched, accompanyChatUser) # 排除陪聊用户的待匹配用户
        if matcheUserExcludeAccompany:
            #普通用户推荐
            toBeMatch = self.getNormalRec(matcheUserExcludeAccompany, self.recLen)
            recResult.extend(toBeMatch)
            if len(recResult) < toBeMatch:
                accompanyUserRec = self.redis.zset_zrangebyscore(self.accompanyUserRecSort, 0, currentTime, True)['result'] #拿出陪聊用户的历史匹配次序
                toBeMatchedAccompany = self.listIntersection(accompanyChatUser, toBeMatched)
                toBeMatch = self.accompanyRec(toBeMatchedAccompany, accompanyUserRec)
                recResult.extend(toBeMatch)
                print toBeMatch
            else:
                #推荐列表长度足够
                pass
        else:
            #没有普通用户，推荐陪聊用户
            accompanyUserRec = self.redis.zset_zrangebyscore(self.accompanyUserRecSort, 0, currentTime, True)['result'] #拿出陪聊用户的历史匹配次序
            toBeMatchedAccompany = self.listIntersection(accompanyChatUser, toBeMatched)
            toBeMatch = self.accompanyRec(toBeMatchedAccompany, accompanyUserRec)
            recResult.extend(toBeMatch)

        return recResult 

    def makeOldIDRec(self): #已测试
        recResult = []
        accompanyChatUser = self.getAccompanyChatUser()
        print "accompanyChatUser " + str(accompanyChatUser)
        currentTime = int(time.time())
        if self.mesh_id in accompanyChatUser:
            return recResult
        if self.tag is None:
            toBeMatched = self.getToBeMatchedUser(self.onlineUserKey ,currentTime)
            print "makeOldIDRec + toBeMatched"
            print toBeMatched
            if toBeMatched:
                recResult = self.makeOldIDRecDetail(toBeMatched, currentTime, accompanyChatUser)
            else:
                #匹配结果为空
                pass

        else:
            #存在暗号的用户
            tagKey = self.onlineUserTagKeyPrex + self.tag
            toBeMatched = self.getToBeMatchedUser(tagKey, currentTime)
            if toBeMatched:
                recResult = self.makeOldIDRecDetail(toBeMatched, currentTime, accompanyChatUser)
            else:
                #匹配结果为空
                pass

        return recResult

    def rec(self):
        if self.ifNewID():
            recResult = self.makeNewIDRec()
            if recResult:              
                return recResult
            else:
                #没有在线用户的时候，返回空
                return []
        else:
            recResult = self.makeOldIDRec()

        self.close()
        return recResult

    def close(self):
        self.redis.release()

if __name__ == '__main__':
    mesh_id = '97836494'
    #mesh_id = '11984388'
    #mesh_id = '84509469'
    #tag = 'abc'
    tag = None
    import datetime
    print datetime.datetime.now()
    rec = Recommend(mesh_id, tag)
    result = rec.rec()
    print result
    print datetime.datetime.now()

