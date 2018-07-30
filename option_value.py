# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 13:39:05 2018

@author: Administrator
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from pyecharts import Line, Grid, Bar, Overlap

plt.rcParams['font.sans-serif'] = ['SimHei']

Term=2
Fixedassets_yield=5.50*Term
fees1=0.10*Term
fees2=0.40*Term
fees3=0.05*Term
fees4=0.00*Term
required_yield=2*Term
option_price=2.30*Term

fund_last_price_df=pd.read_csv('C:/Users/Administrator/Desktop/fof_option/fund_last_price_df.csv',parse_dates=True,index_col=0)
#定义计算实际期权购买值函数
def option_money(Fixedassets_yield,fees1,fees2,fees3,fees4,required_yield):

    option_money=Fixedassets_yield-fees1-fees2-fees3-fees4-required_yield
    return option_money
option_money(Fixedassets_yield,fees1,fees2,fees3,fees4,required_yield)
#定义参与率函数
def option_shares(option_money,option_price):
# =============================================================================
#     option_price:期权费  eg:2.30%
#     option_shares=option_money/option_price
# =============================================================================
    option_shares=option_money(Fixedassets_yield,fees1,fees2,fees3,fees4,required_yield)/option_price
    return option_shares
option_shares(option_money,option_price)
#计算期权收益率
option_yield_pershare=[]
for i in range(252*Term,len(fund_last_price_df)):
    option_yield_pershare.append(np.max([0,fund_last_price_df.values[i]/fund_last_price_df.values[i-252*Term]-1]))
option_yield_pershare=np.multiply(option_yield_pershare,100)
option_yield=np.multiply(option_shares(option_money,option_price),option_yield_pershare)

#all_yield=np.multiply(option_shares(option_money,option_price),option_yield)+required_yield
all_yield=option_yield+required_yield
annual_yield=100*pow((all_yield+100)/100,1/Term)-100
annual_yield_df=pd.DataFrame(np.array(annual_yield).T,index=fund_last_price_df.index[0:len(fund_last_price_df)-252*Term])
all_yield_df=pd.DataFrame(np.array(all_yield).T,index=fund_last_price_df.index[0:len(fund_last_price_df)-252*Term])

#输出收益率统计量
print('收益率的均值为:'+str(all_yield.mean()))
print('收益率的方差为:'+str(all_yield.var()))
print('收益率的最小值为:'+str(all_yield.min()))
print('收益率的最大值为:'+str(all_yield.max()))
#绘制产品收益率分布条形图
fig=plt.figure(figsize=(12,6)) 
count1=count2=count3=count4=count5=count6=count7=count8=count9=count10=count11=count12=0
for i in range(len(annual_yield)):
    if annual_yield[i]==2:
        count1=count1+1
    elif annual_yield[i]>2 and annual_yield[i]<=4:
        count2=count2+1
    elif annual_yield[i]>4 and annual_yield[i]<=6:
        count3=count3+1
    elif annual_yield[i]>6 and annual_yield[i]<=8:
        count4=count4+1
    elif annual_yield[i]>8 and annual_yield[i]<=10:
        count5=count5+1
    elif annual_yield[i]>10 and annual_yield[i]<=12:
        count6=count6+1
    elif annual_yield[i]>12 and annual_yield[i]<=14:
        count7=count7+1
    elif annual_yield[i]>14 and annual_yield[i]<=16:
        count8=count8+1
    elif annual_yield[i]>16 and annual_yield[i]<=18:
        count9=count9+1
    elif annual_yield[i]>18 and annual_yield[i]<=20:
        count10=count10+1
    elif annual_yield[i]>20 and annual_yield[i]<=22:
        count11=count11+1
    else:
        count12=count12+1
count=[count1,count2,count3,count4,count5,count6,count7,count8,count9,count10,count11,count12]
pb=[c/len(annual_yield) for c in count]   
X=[2,4,6,8,10,12,14,16,18,20,22,24]
Y=pb
plt.bar(X,Y,label='option yield distruibution')
plt.legend()
plt.title(u'产品收益率分布图')
plt.savefig('option yield distruibution')
#绘制call option
def sgn(x):
    if x>0:
        return option_shares(option_money,option_price)*x+required_yield/100
x2_max=annual_yield_df.max()/100/option_shares(option_money,option_price)
x2=pylab.linspace(0,x2_max,100)
y2=[]
for i in x2:
    y2.append(sgn(i))
y2[0]=required_yield/100
fig=plt.figure(figsize=(9,6)) 
max_index=99
min_index=0
median_index=50
x1=[-0.1,-0.05,0]
y1=[required_yield/100]*3
plt.plot(x1,y1,color='blue')
plt.plot(x2,y2,color='blue',label='call option')
plt.plot(x2[max_index],y2[max_index],'ks')
show_max='['+'max:'+str(round(x2[max_index],2))+' '+str(round(y2[max_index],2))+']'
plt.annotate(show_max,xytext=(x2[max_index],y2[max_index]),xy=(x2[max_index],y2[max_index]))
plt.plot(x2[min_index],y2[min_index],'ks')
show_min='['+'min:'+str(round(x2[min_index],2))+' '+str(round(y2[min_index],2))+']'
plt.annotate(show_min,xytext=(x2[min_index],y2[min_index]),xy=(x2[min_index],y2[min_index]))
plt.plot(x2[min_index],y2[min_index],'ks')
show_median='['+'median:'+str(round(x2[median_index],2))+' '+str(round(y2[median_index],2))+']'
plt.annotate(show_median,xytext=(x2[median_index],y2[median_index]),xy=(x2[median_index],y2[median_index]))
plt.legend()
plt.title(u'产品收益图')
plt.savefig('call option')
annual_yield_df.describe()
#对未来Term年的收益进行预测
option_yield_predict=[]
for i in range(len(fund_last_price_df)-Term*252,len(fund_last_price_df)):
    option_yield_predict.append(np.max([0,fund_last_price_df.values[-1]/fund_last_price_df.values[i]-1]))
option_yield_predict=np.multiply(option_yield_predict,100)
option_yield_predict=np.multiply(option_shares(option_money,option_price),option_yield_predict)


all_yield_predict=option_yield_predict+required_yield
annual_yield_predict=100*pow((all_yield_predict+100)/100,1/Term)-100
annual_yield_predict_df=pd.DataFrame(np.array(annual_yield_predict).T,index=fund_last_price_df.index[-252*Term-1:-1])
all_yield_predict_df=pd.DataFrame(np.array(all_yield_predict).T,index=fund_last_price_df.index[-252*Term-1:-1])
annual_yield_predict_df[annual_yield_predict_df<2]=2

#波动率
total_std=annual_yield_df.std()
day_std=annual_yield_df.std()/np.sqrt(252)

#加上时间价值的期权收益
all_yield_predict2_df=pd.DataFrame(np.array(all_yield_predict).T,index=fund_last_price_df.index[-252*Term-1:-1])
for i in range(0,252*Term):
    all_yield_predict2_df.values[i]=all_yield_predict_df.values[i]+i*annual_yield_df.mean()/252
annual_yield_predict2_df=100*pow((all_yield_predict2_df+100)/100,1/Term)-100
annual_yield_predict2_df[annual_yield_predict2_df<2]=2
cum_std_up=[]
for t in range(0,252*Term):
    cum_std_up.append(day_std*np.sqrt(t))

predict_yield_up=[]
for i in range(len(cum_std_up)):
    predict_yield_up.append(annual_yield_predict2_df.values[i]+cum_std_up[i])
predict_yield_up_df=pd.DataFrame(predict_yield_up, index=fund_last_price_df.index[-252*Term-1:-1])

cum_std_down=[]
for t in range(0,252*Term):
    cum_std_down.append(day_std*np.sqrt(t))
predict_yield_down=[]
for i in range(len(cum_std_up)):
    predict_yield_down.append(annual_yield_predict2_df.values[i]-cum_std_down[i])
predict_yield_down_df=pd.DataFrame(predict_yield_down, index=fund_last_price_df.index[-252*Term-1:-1])
predict_yield_down_df[predict_yield_down_df<2]=2



cum_std2_up=[]
for t in range(0,252*Term):
    cum_std2_up.append(2*day_std*np.sqrt(t))
# =============================================================================
# for i in range(0,252*Term):
#     cum_std2_up[i]=100*pow((cum_std2_up[i]+100)/100,1/Term)-100
# =============================================================================

cum_std2_down=[]
for t in range(0,252*Term):
    cum_std2_down.append(2*day_std*np.sqrt(t))
# =============================================================================
# for i in range(0,252*Term):
#     cum_std2_down[i]=100*pow((cum_std2_down[i]+100)/100,1/Term)-100
# =============================================================================

predict_yield2_up=[]
for i in range(len(cum_std2_up)):
    predict_yield2_up.append(annual_yield_predict2_df.values[i]+cum_std2_up[i])
predict_yield2_up_df=pd.DataFrame(predict_yield2_up, index=fund_last_price_df.index[-252*Term-1:-1])
predict_yield2_down=[]
for i in range(len(cum_std2_up)):
    predict_yield2_down.append(annual_yield_predict2_df.values[i]-cum_std2_down[i])
predict_yield2_down_df=pd.DataFrame(predict_yield2_down, index=fund_last_price_df.index[-252*Term-1:-1])
#predict_yield_down_df=100*pow((predict_yield_down_df+100)/100,1/Term)-100

predict_yield2_down_df[predict_yield2_down_df<2]=2
#绘图
fig=plt.figure(figsize=(12,6)) 
ax1=fig.add_subplot(111)
plt.xticks(pd.date_range(fund_last_price_df.index[0],fund_last_price_df.index[-1]))#时间间隔 
p1,=ax1.plot(fund_last_price_df,label='FOF_value')
p1_legend = plt.legend(handles=[p1], loc=1)
ax = plt.gca().add_artist(p1)
#设置双坐标轴，右侧Y轴 
ax2=ax1.twinx() 
#设置右侧Y轴显示百分数 
from matplotlib.ticker import FuncFormatter
ylim=int(np.max(annual_yield_df))+5
ax2.set_ylim(0,ylim)
p2,=ax2.plot(annual_yield_df,'r',label='option_yield')
ax2.plot(annual_yield_predict_df,linestyle='--',color='r')
ax2.plot(annual_yield_predict2_df,linestyle='--',color='r')
ax2.plot(predict_yield_up_df,linestyle='--',color='r')
ax2.plot(predict_yield_down_df,linestyle='--',color='r')
ax2.plot(predict_yield2_up_df,linestyle='--',color='black')
ax2.plot(predict_yield2_down_df,linestyle='--',color='black')
plt.legend(handles=[p2], loc=2)
def to_percent(temp, position):
    return '%1.0f'%(temp) + '%'
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.title(u'FOF组合净值和期权收益图')
plt.savefig('FOF&option yield')
plt.show()





dfvalue1=[round(i[0],4) for i in fund_last_price_df.values]
dfvalue2=[round(i[0],4) for i in annual_yield_df.values]
# =============================================================================
# dfvalue3=[i[0] for i in annual_yield_predict_df.values]
# dfvalue4=[i[0] for i in annual_yield_predict2_df.values]
# dfvalue5=[i[0] for i in predict_yield_up_df.values]
# dfvalue6=[i[0] for i in predict_yield_down_df.values]
# dfvalue7=[i[0] for i in predict_yield2_up_df.values]
# dfvalue8=[i[0] for i in predict_yield2_down_df.values]
# =============================================================================

_index=[i for i in fund_last_price_df.index.format()]
_index2=_index[0:len(_index)-252*Term]
_index3=_index[-252*Term:]
line1=Line("fof净值图",title_top="50%")
line1.add("fof", _index, dfvalue1,is_label_show=True,is_more_utils=True, is_fill=True, area_color='#000',
         area_opacity=0.3, is_smooth=True,yaxis_force_interval=0.06,yaxis_interval=10, yaxis_max=1.25, yaxis_min=0.95,legend_top="50%")
# =============================================================================
# line2=Line()
# line2.add("option_yield", _index2, dfvalue2, yaxis_formatter="%", yaxis_interval=10 )
# line3=Line()
# line3.add("option_yield_predict", _index3,dfvalue3, yaxis_formatter="%")
# line3.add("option_yield_predict2", _index3,dfvalue4, yaxis_formatter="%", yaxis_interval=10)
# line3.add("option_yield_up1std", _index3,dfvalue5, yaxis_formatter="%", yaxis_interval=10)
# line3.add("option_yield_down1std", _index3,dfvalue6, yaxis_formatter="%", yaxis_interval=10)
# line3.add("option_yield_up2std", _index3,dfvalue7, yaxis_formatter="%", yaxis_interval=10)
# line3.add("option_yield_down2std", _index3,dfvalue8, yaxis_formatter="%", yaxis_interval=10)
# =============================================================================

# =============================================================================
# overlap=Overlap()
# overlap.add(line1)
# overlap.add(line2, is_add_yaxis=True, yaxis_index=1)
# overlap.add(line3,is_add_yaxis=True, yaxis_index=1)
# 
# =============================================================================
bar=Bar("收益率分布图", height=720)
attr=[2,4,6,8,10,12,14,16,18,20,22,24]
v1=pb
bar.add("收益率分布图", attr, v1,)

grid = Grid()
grid.add(bar, grid_bottom="60%")
grid.add(line1, grid_top="60%")
grid.render()








