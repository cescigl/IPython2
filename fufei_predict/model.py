
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd

import os
import time
import datetime

from redshiftpool import RedShift
from mysqlpool import Mysql


# In[3]:

redshift = RedShift()
#mysql = Mysql()


# In[4]:

#sql_train_neg = "select c.token,to_char(client_time,'YYYY-MM-DD HH24:MI:SS'),c.sta_key,to_char(create_time,'YYYY-MM-DD HH24:MI:SS') from (select b.* from (select token from sta_new_user where server_time>='2017-06-01' and server_time<'2017-06-06' and product_id=600027) a ,(select * from (select token,client_time,sta_key,ROW_NUMBER () OVER (PARTITION BY token order by client_time asc) as row from sta_event_game_publish where server_time>='2017-06-01' and server_time<'2017-06-11' and product_id=600027 and substring(sta_key,1,1)='T') t where row<=30) b where a.token=b.token) c left join (select token,min(create_time) as create_time from game_iap where create_time>='2017-06-01' and product_id=600027 group by token)d on c.token=d.token where d.token is null;"

sql_train_neg = "select c.token,to_char(client_time,'YYYY-MM-DD HH24:MI:SS'),c.sta_key,to_char(create_time,'YYYY-MM-DD HH24:MI:SS') from (select b.* from (select token from sta_new_user where server_time>='2017-07-01' and server_time<'2017-07-06' and product_id=600027) a ,(select * from (select token,client_time,sta_key,ROW_NUMBER () OVER (PARTITION BY token order by client_time asc) as row from sta_event_game_publish where server_time>='2017-07-01' and server_time<'2017-07-11' and product_id=600027 and substring(sta_key,1,1)='T') t where row<=30) b where a.token=b.token) c left join (select token,min(create_time) as create_time from game_iap where create_time>='2017-07-01' and product_id=600027 group by token)d on c.token=d.token where d.token is null;"


sql_train_pos = "select c.token,to_char(client_time,'YYYY-MM-DD HH24:MI:SS'),c.sta_key,to_char(create_time,'YYYY-MM-DD HH24:MI:SS') from (select b.* from (select token from sta_new_user where server_time>='2017-03-01' and product_id=600027) a ,(select * from (select token,client_time,sta_key,ROW_NUMBER () OVER (PARTITION BY token order by client_time asc) as row from sta_event_game_publish where server_time>='2017-03-01' and product_id=600027 and substring(sta_key,1,1)='T') t where row<=30) b where a.token=b.token) c join (select token,min(create_time) as create_time from game_iap where create_time>='2017-04-01' and product_id=600027 group by token)d on c.token=d.token;"



# In[5]:

def getRawData(s,filename):
    raw = redshift.getAll(s)
    fileObject = open(filename, 'w')  
    for r in raw:  
        fileObject.write(str(r))  
        fileObject.write('\n')  
    fileObject.close() 
    return raw


# In[6]:

#raw_train_neg = getRawData(sql_train_neg,'raw_train_neg.txt')
raw_train_pos = getRawData(sql_train_pos,'raw_train_pos.txt')


# In[7]:

raw_train_neg = getRawData(sql_train_neg,'raw_train_neg.txt')


# In[10]:

#公共方法
def getDic(data):
    positiveDataDic={} #是一个储存正样本的字典
    negativeDataDic={} #是一个储存负样本的字典
    for i in range(len(data)):
        key=data[i][0]
        value = []
        value.extend(data[i][1:3])
        if data[i][3]:
            if positiveDataDic.has_key(key):
                positiveDataDic[key].extend([value])
            else:
                positiveDataDic[key]=[value]
        else:
            if negativeDataDic.has_key(key):
                negativeDataDic[key].extend([value])
            else:
                negativeDataDic[key]=[value]

    return positiveDataDic,negativeDataDic


# In[11]:

print raw_train_neg[-10:]
dic_train_neg_pos,dic_train_neg_neg = getDic(raw_train_neg)
dic_train_pos_pos,dic_train_pos_neg = getDic(raw_train_pos)


# In[12]:

keys=['GIB','T01','T02','T03','T04','T05','T06','T07','T08','T09','T0F','T0G','T0H','T0I','T0J','T0K','T0L','T0M','T0O','T0P','T0R','T0S','T0T','T0U','T0V','T0N']

key_indices = dict((key, i) for i, key in enumerate(keys))
indices_key = dict((i, key) for i, key in enumerate(keys))


# In[13]:

def makeTrain(dic):
    result = []
    for (k,v) in dic.items():
        if len(v)<21:
            continue
        else:
            v.sort()
            value = [key_indices[v[i][1]] for i in range(1,21)]
            result.append(value)
    
    return result    


# In[14]:

pos = makeTrain(dic_train_pos_pos)
neg = makeTrain(dic_train_neg_neg)

print len(pos)
print len(neg)

print pos[0:3]
print neg[0:3]


# In[15]:

import random
import copy

random.shuffle(neg)

tlen = 6500

trainX = copy.deepcopy(pos[0:tlen])
trainX.extend(neg[0:tlen])
trainY = np.ones(tlen)
trainY = np.append(trainY, np.zeros(tlen))

testX = copy.deepcopy(pos[tlen:])
leng = len(testX)
testY = np.ones(leng)

print leng
print len(testX),len(testY)
testX.extend(neg[tlen:tlen+leng])
testY = np.append(testY, np.zeros(leng))

print len(trainX),len(trainY)
print len(testX),len(testY)
print testY.shape

print trainY


# In[16]:

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization

maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32


# In[17]:

trainX = np.array(trainX)
testX = np.array(testX)


# In[18]:

print('Build model...')
model = Sequential()
model.add(Embedding(26, 128))
model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.2, return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.2))
model.add(BatchNormalization())
model.add(Dense(64,activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# In[19]:

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(trainX, trainY,
          batch_size=batch_size,
          epochs=20,
          shuffle=True,
          validation_data=(testX, testY))


# In[20]:

import time
print time.time()
result = model.predict(trainX)
print time.time()


# In[21]:

#验证训练数据
print len(result)
posl = result[:tlen]
negl = result[tlen:]

posloc = np.where(posl>=0.35)
negloc = np.where(negl>=0.35)
print len(posloc[0])
print len(negloc[0])


posloc = np.where(posl>=0.45)
negloc = np.where(negl>=0.45)

print len(posloc[0])
print len(negloc[0])


# In[22]:

print result


# In[23]:

resultpos = model.predict(np.array(pos))
resultneg = model.predict(np.array(neg))


# In[28]:

rate=0.2

posloc = np.where(resultpos>=rate)
negloc = np.where(resultpos<=rate)
print len(resultpos)
print len(posloc[0])
print "分到正类百分比：",len(posloc[0])*1.0/len(resultpos)

posloc = np.where(resultneg>=rate)
negloc = np.where(resultneg<=rate)
print len(resultneg)
print len(posloc[0])
print "分到负类百分比：",len(negloc[0])*1.0/len(resultneg)


