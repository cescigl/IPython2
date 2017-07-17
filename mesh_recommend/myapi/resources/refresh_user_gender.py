# -*- coding: utf-8 -*-

import redis
import json 
import sys
sys.path.append("..")
from common.config import RedisDBConfig
r = redis.Redis(host=RedisDBConfig.HOST, port=RedisDBConfig.PORT)

userGenderMaleMale = 'rec_user_gender_male_male' # 保存男性用户愿意匹配男性用户的列表
userGenderMaleFemale = 'rec_user_gender_male_female' # 保存男性用户愿意匹配女性用户的列表
userGenderFemaleMale = 'rec_user_gender_female_male' # 保存女性用户愿意匹配男性用户的列表
userGenderFemaleFemale = 'rec_user_gender_female_female' # 保存女性用户愿意匹配女性用户的列表

userDetail = '_tt_User' #保存用户的所有属性， hset

user = r.hgetall(userDetail)

for key in user:
    userDesDict = json.loads(user[key])
    gender = userDesDict['gender'] if userDesDict.has_key('gender') else 2
    meshgender = userDesDict['matchGender'] if userDesDict.has_key('matchGender') else 100 #如果没有确定的meshgender信息，则取100
    print key
    if gender == 0:
        if meshgender == 101:
            matchGender = r.zrange(userGenderFemaleFemale, 0, -1)
            r.zadd(userGenderFemaleFemale, key, 0)
            r.zrem(userGenderMaleFemale, key)
            r.zrem(userGenderFemaleMale, key)
            r.zrem(userGenderMaleMale ,key)
        elif meshgender == 102:
            matchGender = r.zrange(userGenderMaleFemale, 0, -1)
            r.zadd(userGenderFemaleMale, key, 0)
            r.zrem(userGenderMaleFemale, key)
            r.zrem(userGenderMaleMale ,key)
            r.zrem(userGenderFemaleFemale, key)
        else:
            matchGender = r.zrange(userGenderFemaleFemale, 0, -1)
            matchGender.extend(r.zrange(userGenderMaleFemale, 0, -1))
            r.zadd(userGenderFemaleFemale, key, 0)
            r.zadd(userGenderFemaleMale, key, 0)
            r.zrem(userGenderMaleFemale, key)
            r.zrem(userGenderMaleMale ,key)

    elif gender == 1:
        if meshgender == 101:
            matchGender = r.zrange(userGenderFemaleMale, 0, -1)
            r.zadd(userGenderMaleFemale, key, 0)
            r.zrem(userGenderFemaleMale, key)
            r.zrem(userGenderMaleMale ,key)
            r.zrem(userGenderFemaleFemale, key)
        elif meshgender == 102:
            matchGender = r.zrange(userGenderMaleMale, 0, -1)
            r.zadd(userGenderMaleMale, key, 0)
            r.zrem(userGenderFemaleMale, key)
            r.zrem(userGenderFemaleFemale, key)
            r.zrem(userGenderMaleFemale, key)
        else:
            matchGender = r.zrange(userGenderFemaleMale, 0, -1)
            matchGender.extend(r.zrange(userGenderMaleMale, 0, -1))
            r.zadd(userGenderMaleFemale, key, 0)
            r.zadd(userGenderMaleMale, key, 0)
            r.zrem(userGenderFemaleMale, key)
            r.zrem(userGenderFemaleFemale, key)
    else:
        pass

