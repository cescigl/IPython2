# -*- coding: UTF-8 -*-
"""
config是一些配置信息
jdbc:redshift://mesh-data.ciemeqgcfm18.us-west-2.redshift.amazonaws.com:5439/mesh
datauser / FyBJsoApvby8NWqX
"""
class RedShiftConfig:
    HOST = 'sta-event-game-publish.ciemeqgcfm18.us-west-2.redshift.amazonaws.com'
    PORT = 5439
    USER = 'analysis'
    PASSWORD = 'D2Woy6crKqGDszId'
    DATABASE = 'event'


"""
import redis
r = redis.Redis(host='172.16.0.192', port=6379)
r = redis.Redis(host='prod-2mesh-001.hzy1hi.0001.usw2.cache.amazonaws.com', port=6379)
r.zadd('rec_test_zhaihuixin','a',1)
r.zadd('rec_test_zhaihuixin','b',2)
r.zadd('rec_test_zhaihuixin','c',1.5)
r.zrange('rec_test_zhaihuixin',0,-1)
"""
class RedisDBConfig:
    HOST = 'prod-2mesh-001.hzy1hi.0001.usw2.cache.amazonaws.com'
    PORT = 6379
    DBID = 0
    MC = 500

"""
Config是一些数据库的配置文件
db.url=jdbc:mysql://test-holagames.cnurzh37opih.us-west-2.rds.amazonaws.com:3306/mesh?useunicode=true&characterEncoding=UTF-8&autoReconnect=true
db.driver=com.mysql.jdbc.Driver
db.user=mesh
db.pass=r76dG6CbpbBvt5I7MH0T
"""

class MysqlConfig:
    DBHOST = "test-dataverse-web.c0poh9vgjxya.rds.cn-north-1.amazonaws.com.cn"
    DBPORT = 3306
    DBUSER = "datauser"
    DBPWD = "ebOUeWSin3hAAKKD"
    DBNAME = "dataverse"
    DBCHAR = "utf8"


class DDBConfig:
    HOST = 'internal-internal-zookeeper-oregon-1843879134.us-west-2.elb.amazonaws.com:21810'
    ZK_KEY = '/ba45fa7423cb83a963d93735b445fd84'
    REGION = 'us-west-2'
