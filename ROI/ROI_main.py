# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import datetime
import ROI_model


def main():
    if len(sys.argv) < 2:
       print "Please write table name!"
       exit() 
    
    tableName = sys.argv[1]
    print tableName
    colname = ''
    if len(sys.argv) == 3:
        colname = sys.argv[2]
    print colname
    today = datetime.date.today().strftime("%Y%m%d")
    print "begin"
    
    #预测天数
    predictDays = 90
    
    
    ROI_model.entrance(tableName, predictDays, colname)


if __name__ == '__main__':
    main()
