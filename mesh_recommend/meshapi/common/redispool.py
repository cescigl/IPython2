# -*- coding: UTF-8 -*-
import redis
from config import RedisDBConfig


def operator_status(func):
    '''get operatoration status
    '''
    def gen_status(*args, **kwargs):
        error, result = None, None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            error = str(e)

        return {'result': result, 'error':  error}

    return gen_status

class RedisCache(object):
    def __init__(self):
        if not hasattr(RedisCache, 'pool'):
            print "create connect redis"
            RedisCache.create_pool()
        self._connection = redis.Redis(connection_pool = RedisCache.pool)

    @staticmethod
    def create_pool():
        RedisCache.pool = redis.ConnectionPool(
                host = RedisDBConfig.HOST,
                port = RedisDBConfig.PORT,
                db   = RedisDBConfig.DBID,
                max_connections = RedisDBConfig.MC)

    @operator_status
    def set_data(self, key, value):
        '''set data with (key, value)
        '''
        return self._connection.set(key, value)

    @operator_status
    def get_data(self, key):
        '''get data by key
        '''
        return self._connection.get(key)

    @operator_status
    def del_data(self, key):
        '''delete cache by key
        '''
        return self._connection.delete(key)

    @operator_status
    def set_expire(self, key, times):
        '''set expire times
        '''
        return self._connection.expire(key, times)
    @operator_status
    def zset_zadd(self, key, value, score):
        return self._connection.zadd(key, value, score)

    @operator_status
    def zset_zrem(self, key, value):
        return self._connection.zrem(key, value)

    @operator_status
    def zset_zrange(self, key, begin, end):
        return self._connection.zrange(key, begin, end)

    @operator_status
    def zset_zrangebyscore(self, key, min, max, ws=False):
        return self._connection.zrangebyscore(key, min, max, withscores=ws) 

    @operator_status
    def hset_hget(self, name, key):
        return self._connection.hget(name, key)

    @operator_status
    def hset_hset(self, name, key, value):
        return self._connection.hset(name, key, value)

    @operator_status
    def set_smembers(self, name):
        return self._connection.smembers(name)

    def release(self):
        del self._connection


if __name__ == '__main__':
    print RedisCache().set_data('zhaihuixin_Testkey', "Simple Test")
    print RedisCache().get_data('zhaihuixin_Testkey')
    print RedisCache().del_data('zhaihuixin_Testkey')
    print RedisCache().get_data('zhaihuixin_Testkey')
    RedisCache().release()
