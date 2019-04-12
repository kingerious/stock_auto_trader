import pandas as pd
import tushare as ts
import numpy as np
from sklearn.cross_validation import KFold  # For K-fold cross validation
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from twilio.rest import Client
import datetime
import time
import sys
sys.path.extend(['/Users/kingerious/git/stockPredictor'])
import hola

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


def getTarget(filterCodesList, today_all):
    max_score, sumScore, sumCount, max_code = 0, 0, 0, ''
    high_rate_list = []
    for code in filterCodesList:
        x_today, df = hola.getCodeDetail(code, today_all)
        print(x_today)
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
                high_rate_list.append([code, x_today[3]])

    if high_rate_list:
        high_rate_list = sorted(high_rate_list, key=lambda x:x[1])
        high_rate_list = [i[0] for i in high_rate_list]
    print(high_rate_list)
    print("finished, max_score = %.3f, code: %s, aveScore = %.3f" % (max_score, max_code, sumScore/sumCount))
    try:
        sendMessage(high_rate_list)
    except:
        pass
    return high_rate_list[len(high_rate_list)//2]

def sendMessage(high_rate_list):
    # 下面认证信息的值在你的 twilio 账户里可以找到
    account_sid = "ACd1516a7bc5a58c888c064154bff19389"
    auth_token = "0af7b188b6900fdd848c52e6b58d8c9d"
    client = Client(account_sid, auth_token)
    content = ','.join(high_rate_list)
    # noticePhone = ["+8618582557010", "+8618081723936"]
    noticePhone = ["+8618582557010"]
    for phone in noticePhone:
        message = client.messages.create(to=phone,  # 区号+你的手机号码
                                         from_="+12015145420",  # 你的 twilio 电话号码
                                         body=content)
        print(message.sid)

def getFilterStock():
    print(datetime.datetime.today())
    sh, sh_real_time = ts.get_hist_data('sh'), ts.get_index()
    if sh_real_time.iloc[0]['change'] < -0.8 or (
            sh_real_time.iloc[0]['close'] < sh.iloc[4]['close'] and sh_real_time.iloc[0]['change'] < 0.8):
        print("大盘不好，不操作")
        sys.exit()
    while 1:
        try:
            df = ts.get_today_all()
            today_all = df[~df.name.str.contains('ST')]
            break
        except:
            print("正在重试.")
    filterCodesList = hola.getFilterCodesList(today_all)
    return filterCodesList

def main():
    filterCodesList = getFilterStock()
    # filterCodesList = ['603888', '603128', '603117', '601800', '601700', '601668', '601099', '601009', '600975', '600868', '600848', '600804', '600795', '600737', '600705', '600643', '600459', '600446', '600406', '600369', '600292', '600218', '600192', '600166', '600158', '600131', '600061', '300435', '300333', '300297', '300113', '300091', '002733', '002673', '002639', '002596', '002477', '002451', '002396', '002385', '002366', '002299', '002235', '002130', '002047', '002002', '001696', '000990', '000957', '000930', '000875', '000760', '000735', '000625', '000571', '000563', '000425', '000338']
    while datetime.datetime.now().hour != 14 or datetime.datetime.now().minute != 30:
       time.sleep(30)
    while 1:
        try:
            df = ts.get_today_all()
            today_all = df[~df.name.str.contains('ST')]
            break
        except:
            print("正在重试.")
    return getTarget(filterCodesList, today_all)


if __name__ == '__main__':
    main()
