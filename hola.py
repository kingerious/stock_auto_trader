import tushare as ts
import pandas as pd

pd.set_option('display.max_column', 1000)
pd.set_option('display.width', 1000)


def stockFilter(codesList, today_all):
    filterCodesList = []
    print("需遍历%d只股票" % len(codesList))
    black_stock = ['300684']
    offset = 1
    for i in codesList:
        try:
            currentStock = ts.get_hist_data(i)
            todayStock = today_all[today_all['code'] == i]
            if currentStock.iloc[0]['ma10'] > currentStock.iloc[1]['ma10'] and currentStock.iloc[0]['ma20'] > currentStock.iloc[1]['ma20'] \
                    and float(todayStock['amount']) > 1000000000 and currentStock.shape[0] > 400 and float(todayStock['trade']) != 0.0\
                    and currentStock.iloc[0]['ma5'] > currentStock.iloc[1]['ma5'] and i not in black_stock\
                    and -9.9 < float(todayStock['changepercent']) < 9.9:
                filterCodesList.append(i)
                print("遍历第%d只，filterCodesList 中添加%s" % (offset, i))
        except:
            print("no data")
        finally:
            # print("filterCodesList 中添加%s" % (i))
            offset += 1
    return filterCodesList


def getFilterCodesList(today_all):
    codesList = today_all.loc[:, 'code'].values
    filterCodesList = stockFilter(codesList, today_all)
    print(filterCodesList)
    return filterCodesList


def getFilterCodesListNew(today_all):
    pro = ts.pro_api('dbeea958a3553c07611d0b60bc4424f7b901355d113a3b97262e5d3f')
    stock_list = pro.stock_basic(list_status='L')
    stock_list = list(stock_list['symbol'].values)
    stock_list = stockFilter(stock_list, today_all)
    print(stock_list)
    return stock_list


def getCodeDetail(code, today_all):
    df = ts.get_hist_data(code)
    condition = today_all['code'] == code
    today_code = today_all[condition]
    x_today = [0, 0, 0, 0, 0]
    try:
        x_today_ma5 = (today_code['trade'].values[0] + df['close'].iloc[:4].sum()) / 5 - df.iloc[0]['ma5']
        x_today_v_ma5 = (today_code['volume'].values[0]/100 + df['volume'].iloc[:4].sum()) / 5 - df.iloc[0]['v_ma5']
        x_today = [x_today_ma5, x_today_v_ma5, today_code['volume'].values[0]/100, today_code['changepercent'].values[0], today_code['low'].values[0]]
    except:
        print("get code detail error, code:" + code)
    return x_today, df