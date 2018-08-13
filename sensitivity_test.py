# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 08:43:00 2018

@author: Administrator
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#导入原始数据
fund_last_price_df=pd.read_csv('C:/Users/Administrator/Desktop/fof_option/fund_last_price_df.csv',parse_dates=True,index_col=0)
#敏感性测试
#设置留存收益和固定收益
floor=[0,1,2,3,4,5]
fix=[4,4.5,5,5.5,6,6.5]
#定义参与率函数
def participation(fix,floor,call=2.0):
    participation=[]
    for i in range(len(fix)):
        for j in range(len(floor)):
            participation.append((fix[i]-floor[j])/call)
    return participation
#在这里修改call_option的值
arr1=np.array(participation(fix,floor,call=2))
arr2=arr1.reshape(len(fix),len(floor))
arr2[arr2<0]=None
arr2
df_partici=pd.DataFrame(arr2*100)
#绘制参与率表格
fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])
vals = np.around(df_partici.values,2)
the_table=plt.table(cellText=vals, rowLabels=fix, colLabels=floor, 
                    colWidths = [0.1]*vals.shape[1], loc='center',cellLoc='center')
the_table.set_fontsize(20)
the_table.scale(2.5,2.58)
plt.title('参与率(%)')

#收益率最大值测算
Term=1
call=2.0
option_yield_max=[]
option_yield_mean=[]
option_yield_pershare2=[]
for i in range(len(arr1)):
    for k in range(252*Term,len(fund_last_price_df)):
        option_yield_pershare2.append(np.max([0,fund_last_price_df.values[k]/fund_last_price_df.values[k-252*Term]-1]))
    option_yield_pershare3=np.multiply(option_yield_pershare2,100)
    option_yield2=np.multiply(arr1[i],option_yield_pershare3)
    option_yield_max.append(option_yield2.max())
    option_yield_mean.append(option_yield2.mean())
opmax1=np.array(option_yield_max) 
opmax1=np.array(floor*6)+opmax1
opmax2=opmax1.reshape(len(fix),len(floor))  
annual_opmax=[100*pow((opmax2+100)/100,1/Term)-100 for opmax2 in opmax2]
df_opmax=pd.DataFrame(annual_opmax)
fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])
vals = np.around(df_opmax.values,4)
the_table=plt.table(cellText=vals, rowLabels=fix, colLabels=floor, 
                    colWidths = [0.1]*vals.shape[1], loc='center',cellLoc='center')
the_table.set_fontsize(20)
the_table.scale(2.5,2.58)
plt.title('最大收益（%）')
#收益率均值
opmean1=np.array(option_yield_mean)   
annual_opmean2=[100*pow((opmean1+100)/100,1/Term)-100 for opmean1 in opmean1]
annual_opmean2=annual_opmean2+np.array(floor*6)
annual_opmean2=annual_opmean2.reshape(len(fix),len(floor))
#绘制最大收益表格
df_opmean=pd.DataFrame(annual_opmean2)
fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])
vals = np.around(df_opmean.values,4)
the_table=plt.table(cellText=vals, rowLabels=fix, colLabels=floor, 
                    colWidths = [0.1]*vals.shape[1], loc='center',cellLoc='center',)
plt.title('平均收益（%）')
the_table.set_fontsize(20)
the_table.scale(2.5,2.58)