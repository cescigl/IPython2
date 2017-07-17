# -*- coding: UTF-8 -*-
from boto3.session import Session
from kazoo.client import KazooClient
from boto3.dynamodb.conditions import Key, Attr
from config import DDBConfig


class DDB(object):
    """docstring for DDB"""
    def __init__(self):
        if not hasattr(DDB, 'dynamodb'):
            print "create connect"
            DDB.createCon()
        self._con = DDB.dynamodb

    @staticmethod
    def createCon():
        zk = KazooClient(hosts=DDBConfig.HOST)
        zk.start()
        temp=zk.get(DDBConfig.ZK_KEY)
        aws_id = temp[0].split(':::')[0]
        aws_key = temp[0].split(':::')[1]
        region = DDBConfig.REGION

        session = Session(aws_access_key_id=aws_id,
                  aws_secret_access_key=aws_key,
                  region_name=region)

        DDB.dynamodb = session.resource('dynamodb')

    def getByKey(self, t, key):
        table = self._con.Table(t)
        item = table.get_item(Key=key).get('Item')
        return item

    def scanByAttr(self, t, attr, value):
    	#table.scan(FilterExpression=Attr('eventId').eq('1482815983639'))
    	table = self._con.Table(t)
    	result = table.scan(FilterExpression=Attr(attr).eq(value))
        return result['Items']

if __name__ == '__main__':
    key = {'hFlag':'1', 'rFlag':'585b8b40e4b00870402bbaf8'}
    table = 'prod_tt_Event'

    ddb = DDB()
    print ddb.getByKey(table, key)
    print ddb.scanByAttr(table, 'eventId', '1482815983639')

