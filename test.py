import easytrader

user = easytrader.use('ths')
user.connect(r'D:\Program Files\ths\xiadan.exe')
balance = user.balance
print(balance)