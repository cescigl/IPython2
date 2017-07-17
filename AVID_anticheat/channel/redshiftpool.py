# -*- coding: UTF-8 -*-
import psycopg2
from config import RedShiftConfig


class RedShift(object):
    """docstring for RedShift"""
    def __init__(self):
        if not hasattr(RedShift, 'con'):
            print "create connect redshift"
            RedShift.createCon()
        self._conn = RedShift.con
        self._cursor = self._conn.cursor()

    @staticmethod
    def createCon():
        RedShift.con=psycopg2.connect(database= RedShiftConfig.DATABASE, host=RedShiftConfig.HOST, port= RedShiftConfig.PORT, user= RedShiftConfig.USER, password= RedShiftConfig.PASSWORD)


    def getAll(self,sql,param=None):
        """
        @summary: 执行查询，并取出所有结果集
        @param sql:查询ＳＱＬ，如果有查询条件，请只指定条件列表，并将条件值使用参数[param]传递进来
        @param param: 可选参数，条件列表值（元组/列表）
        @return: result list(字典对象)/boolean 查询到的结果集
        """
        if param is None:
            self._cursor.execute(sql)
        else:
            self._cursor.execute(sql,param)
        result = self._cursor.fetchall()

        return result
 
    def getOne(self,sql,param=None):
        """
        @summary: 执行查询，并取出第一条
        @param sql:查询ＳＱＬ，如果有查询条件，请只指定条件列表，并将条件值使用参数[param]传递进来
        @param param: 可选参数，条件列表值（元组/列表）
        @return: result list/boolean 查询到的结果集
        """
        if param is None:
            self._cursor.execute(sql)
        else:
            self._cursor.execute(sql,param)

        result = self._cursor.fetchone()

        return result
 
 
    def begin(self):
        """
        @summary: 开启事务
        """
        self._conn.autocommit(0)
 
    def end(self,option='commit'):
        """
        @summary: 结束事务
        """
        if option=='commit':
            self._conn.commit()
        else:
            self._conn.rollback()
 
    def dispose(self,isEnd=1):
        """
        @summary: 释放连接池资源
        """
        if isEnd==1:
            self.end('commit')
        else:
            self.end('rollback');
        self._cursor.close()
        self._conn.close()

if __name__ == '__main__':
    sql1 = "select * from mesh_gift limit 2;"
    sql2 = "select * from mesh_block limit 2;"
    #redshift = RedShift()
    #print redshift.getAll(sql1)
    #print redshift.getAll(sql2)

    #print RedShift().getAll(sql1)
    #print RedShift().getAll(sql2)
    #redshift.dispose(1)

    a = RedShift().getAll("select * from meshmatch_user_info limit 5;")
    print a 
    print type(a[0])
