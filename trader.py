import easytrader
import predictor
import datetime
import time
import sys
import tushare as ts

code = predictor.main()
# code = '600218'
print("-----马上开始交易_%s -----" % code)
user = easytrader.use('ths')
user.connect(r'D:\Program Files\ths\xiadan.exe')
balance = user.balance

print(balance['可用金额'])
print("waiting at:%s" % datetime.datetime.today())
while datetime.datetime.now().hour != 14 or datetime.datetime.now().minute != 57:
    time.sleep(30)
while 1:
    try:
        price = ts.get_today_ticks(code).iloc[0]['price']
        break
    except:
        print("正在重试……")
        time.sleep(5)
amount = balance['总资产']//price//100*100
print("交易时间：%s" % datetime.datetime.today())
user.auto_ipo()
user.buy(code, price*1.005, amount)
print("委托买入：%s，委托价：%f，委托数量：%d" % (code, price*1.005, amount))
try:
    predictor.sendMessage("买入" + code + "成功，" + "一共" + str(amount) + "股")
except:
    pass
time.sleep(68400)
# time.sleep(30)
while 1:
    try:
        code_info = ts.get_today_ticks(code)
        if (float(code_info.iloc[0]['pchange']) - float(code_info.iloc[100]['pchange']) > 2 and float(
                code_info.iloc[0]['pchange']) < 9.9) or (
                datetime.datetime.now().hour == 11 and datetime.datetime.now().minute >= 28):
            user.sell(code, code_info.iloc[0]['price'] * 0.995, amount)
            print("委托卖出：%s，委托价：%f，委托数量：%d" % (code, price * 0.995, amount))
            try:
                predictor.sendMessage("卖出" + code + "成功，" + "一共" + str(amount) + "股")
            except:
                pass
            break
    except:
        print("正在重试……")
        time.sleep(5)
        continue
    time.sleep(120)
try:
    predictor.sendMessage('今天操作成功')
except:
    pass
sys.exit()
