import pandas as pd
import tushare as ts
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.cross_validation import KFold  # For K-fold cross validation
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from twilio.rest import Client
import datetime
import sys
sys.path.extend(['/Users/kingerious/PycharmProjects/stockPredictor'])
import main.hola

pd.set_option('display.max_column', 1000)
pd.set_option('display.width', 1000)


# Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome, code):
    # Fit the model:
    model.fit(data[predictors], data[outcome])

    # Make predictions on training set:
    predictions = model.predict(data[predictors])

    # Print accuracy
    print(code + ":")
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    # Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (data[predictors].iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    # Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors], data[outcome])
    return model, "{0:.3%}".format(np.mean(error))


# print(datetime.datetime.date())
# filterCodesList = ['300009', '002813']
max_score = 0
try:
    df = ts.get_today_all()
    today_all = df[~df.name.str.contains('ST')]
except:
    print("Got stock info failed. Try again later please.")
    sys.exit()
# filterCodesList = main.hola.getFilterCodesListNew(today_all)
filterCodesList = ['000001', '000002', '000063', '000333', '000338', '000503', '000717', '000735', '000807', '000858', '000932', '000981', '000990', '002024', '002027', '002110', '002118', '002146', '002157', '002177', '002191', '002405', '002415', '002440', '002456', '002460', '002565', '002597', '002639', '002648', '002797', '300033', '300059', '300107', '300267', '300498', '300527', '600018', '600028', '600030', '600031', '600036', '600048', '600072', '600104', '600109', '600155', '600196', '600309', '600332', '600340', '600352', '600422', '600426', '600446', '600516', '600518', '600519', '600572', '600585', '600596', '600648', '600690', '600801', '600837', '600887', '601166', '601186', '601211', '601318', '601390', '601398', '601555', '601668', '601688', '601766', '601881', '601989', '603000', '603128', '603993']
high_rate_list = []
Candi_list = []
sumScore = sumCount = 0

# try:
#     df = ts.get_today_all()
#     today_all = df[~df.name.str.contains('ST')]
# except:
#     print("got stock info failed, try again later please.")
#     sys.exit()
for code in filterCodesList:
    x_today, df = main.hola.getCodeDetail(code, today_all)
    print(x_today)
    # exit()
    y = df.loc[:, 'p_change']
    x = df.loc[:, ['ma5', 'v_ma5', 'close', 'volume', 'p_change', 'high', 'low', 'open']]
    if x_today == [0, 0, 0, 0, 0] or x_today[3] > 9.9 or x_today[3] < 0:
        continue

    y = y.apply(lambda x:1 if x > 2 else 0)
    # print(x[:50])
    # print(len(x))pip
    i = 0
    ma5_change = []
    v_ma5_change = []
    close_yesterday = []
    volume_yesterday = []
    p_change_yesterday = []
    high_yesterday = []
    low_yesterday = []
    open_yesterday = []
    # x_today = [input("ma5_change"), input("v_ma5_change")]

    while(i < (len(x)-2)):
        ma5_change.append(x.iloc[i+1]["ma5"] - x.iloc[i+2]["ma5"])
        v_ma5_change.append(x.iloc[i+1]["v_ma5"] - x.iloc[i+2]["v_ma5"])
        close_yesterday.append(x.iloc[i+1]['close'])
        volume_yesterday.append(x.iloc[i+1]['volume'])
        p_change_yesterday.append(x.iloc[i + 1]['p_change'])
        high_yesterday.append(x.iloc[i + 1]['high'])
        low_yesterday.append(x.iloc[i + 1]['low'])
        open_yesterday.append(x.iloc[i + 1]['open'])
        i += 1

    if len(ma5_change) < len(x):
        score = 0

        ma5_change.append(0)
        ma5_change.append(0)
        v_ma5_change.append(0)
        v_ma5_change.append(0)
        close_yesterday.append(0)
        close_yesterday.append(0)
        volume_yesterday.append(0)
        volume_yesterday.append(0)
        p_change_yesterday.append(0)
        p_change_yesterday.append(0)
        high_yesterday.append(0)
        high_yesterday.append(0)
        low_yesterday.append(0)
        low_yesterday.append(0)
        open_yesterday.append(0)
        open_yesterday.append(0)

        x['ma5_change'] = ma5_change
        x['v_ma5_change'] = v_ma5_change
        # x['close_yesterday'] = close_yesterday
        x['volume_yesterday'] = volume_yesterday
        x['p_change_yesterday'] = p_change_yesterday
        # x['high_yesterday'] = high_yesterday
        x['low_yesterday'] = low_yesterday
        # x['open_yesterday'] = open_yesterday
        x['result'] = y

        x.pop('ma5')
        x.pop('v_ma5')
        x.pop('close')
        x.pop('volume')
        x.pop('p_change')
        x.pop('high')
        x.pop('low')
        x.pop('open')

        x = x[:-2]

        # sc = StandardScaler()
        # sc.fit(x)
        # x = sc.transform(x)
        # x_today = sc.transform(x_today)
        #
        # x['result'] = y

        outcome_var = "result"
        model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
        predictor_var = ["ma5_change", "v_ma5_change", 'volume_yesterday', 'p_change_yesterday', 'low_yesterday']
        model, score = classification_model(model, x, predictor_var, outcome_var, code)
        sumScore += float(score[:-1])/100
        sumCount += 1
        tomorrow = model.predict([x_today])
        print(tomorrow)
        if float(score[:-1])/100 >= max_score and tomorrow == [1]:
            max_score = float(score[:-1])/100
            max_code = code
        if float(score[:-1])/100 >= 0.56 and tomorrow == [1]:
            high_rate_list.append(code)
print(high_rate_list)

print("finished, max_score = %.3f, code: %s, aveScore = %.3f" % (max_score, max_code, sumScore/sumCount))

# 下面认证信息的值在你的 twilio 账户里可以找到
account_sid = "ACd1516a7bc5a58c888c064154bff19389"
auth_token = "0af7b188b6900fdd848c52e6b58d8c9d"
client = Client(account_sid, auth_token)
# content = '''noticed by kingerious: buy the stock with market price:''' + max_code + ''', winning percentage:''' + str(max_score) + '''! Unsubscribed back to T.'''
content = ','.join(high_rate_list)
# noticePhone = ["+8618582557010", "+8618081723936"]
noticePhone = ["+8618582557010"]
for phone in noticePhone:
    message = client.messages.create(to=phone,  # 区号+你的手机号码
                                 from_="+12015145420",  # 你的 twilio 电话号码
                                 body=content)
    print(message.sid)

sys.exit()
