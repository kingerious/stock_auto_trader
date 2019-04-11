import easytrader
import datetime
import tushare as ts
import pandas as pd

pd.set_option('display.max_column', 1000)
pd.set_option('display.width', 1000)

# print(ts.get_today_ticks('300376').iloc[0]['price']*0.95)

# user = easytrader.use('ths')
# user.connect(r'D:\Program Files\ths\xiadan.exe')
#  balance = user.balance ()
# #dic = dict(balance) 
# print(balance['可用金额']) 
# print(balance) 
#print(user.position) 
#user.sell("300099", 11, 800)
print(ts.get_index().iloc[0]['close'])
sh = ts.get_hist_data('sh')
print(sh)

print(ts.get_index().iloc[0]['close']>sh.iloc[4]['close'])