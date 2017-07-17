#!/bin/bash
########
#负责对mesh匹配情况进行统计
########
cd `dirname $0`
SCRPATH=`pwd`

LOGDIR=$SCRPATH/../log/
DATE=`date +%Y-%m-%d --date="-1 day"`
THREEDAYAGO=`date +%Y-%m-%d --date="-3 day"`
export LANG=en_US.UTF-8
FIVEDAYAGO=`date '+%a %b' --date="-5 day"`
echo $THREEDAYAGO
echo $FIVEDAYAGO

CNT1=`grep $THREEDAYAGO $LOGDIR/mesh_matching_logging.txt | wc -l`
CNT2=`grep "$FIVEDAYAGO" $SCRPATH/../uwsgi.log | wc -l`
if [ $CNT1 -gt 0 ];then
sed -ci '/'"$THREEDAYAGO"'/,$!d' $LOGDIR/mesh_matching_logging.txt
fi
if [ $CNT2 -gt 0 ];then
sed -ci '/'"$FIVEDAYAGO"'/,$!d' $SCRPATH/../uwsgi.log
fi

CNT_ALL=`cat $LOGDIR/mesh_matching_logging.txt | grep $DATE | grep "match result user" | wc -l`
CNT_NONE=`cat $LOGDIR/mesh_matching_logging.txt | grep $DATE | grep "match result user" | grep "\[\]" | wc -l`
CNT_ONE=`cat $LOGDIR/mesh_matching_logging.txt | grep $DATE | grep "match result user" | grep "\['[0-9]\{8\}'\]" | wc -l`
CNT_MANY=`cat $LOGDIR/mesh_matching_logging.txt | grep $DATE | grep "match result user" | grep -v "\['[0-9]\{8\}'\]"|grep -v "\[\]" | wc -l`

echo "总匹配次数|匹配人数为空次数|匹配人数为1次数|匹配人数>=2次数" > $LOGDIR/matchResult.log
echo $CNT_ALL"|"$CNT_NONE"|"$CNT_ONE"|"$CNT_MANY >> $LOGDIR/matchResult.log

$SCRPATH/create_html.sh $LOGDIR/matchResult.log $LOGDIR/matchResult.html Mesh匹配结果分析-$DATE hx.zhai@avid.ly

/usr/sbin/sendmail hx.zhai@avid.ly <$LOGDIR/matchResult.html
/usr/sbin/sendmail jl.wu@avid.ly <$LOGDIR/matchResult.html




