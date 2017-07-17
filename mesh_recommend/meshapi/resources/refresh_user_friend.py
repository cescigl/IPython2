# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import sys
sys.path.append("/data/mesh_push_service/meshapi/")
from common.redshiftpool import RedShift
from common.config import RedisDBConfig
import redis
import logging
import os


class RefreshFriend(object):
    """docstring for RefreshFriend"""
    def __init__(self):
        super(RefreshFriend, self).__init__()
        self.redshift = RedShift()
        self.userFriendPrex = 'rec_user_friend_' #记录用户的好友列表 （set）
        self.log_filename="/data/mesh_push_service/meshapi/log/refresh_user_friend.txt" 
        self.log_format=' [%(asctime)s]   %(message)s' 
        #将日志文件格式化  
        import logging
        logging.basicConfig(format=self.log_format,datafmt='%Y-%m-%d %H:%M:%S %p',level=logging.DEBUG,filename=self.log_filename,filemode='a') 

    def loadFriendData(self):
        """
        从redshift加载好友信息
        """
        sql = "select userid,oppositeuserid from meshmatch_friend_prod;"
        sqlResult = self.redshift.getAll(sql)
        logging.debug('loadFriendData')

        return sqlResult

    def friendInit(self):
        """
        预处理数据
        """
        r = redis.Redis(host=RedisDBConfig.HOST, port=RedisDBConfig.PORT)
        p = r.pipeline()

        friend = self.loadFriendData() #用户好友信息

        for i in friend:
            p.delete(i[0])

        p.execute()
        for i in friend:
            p.sadd(self.userFriendPrex + i[0], i[1])
            p.sadd(self.userFriendPrex + i[1], i[0])
        
        logging.debug('write friend info to redis')


if __name__ == '__main__':
    rf = RefreshFriend()
    rf.friendInit()


