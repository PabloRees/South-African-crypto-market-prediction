import numpy as np
import pandas as pd
import pytz

def Y_cat_format(df,YVar,binary:bool):
    Y_mean = np.mean(df[YVar])

    if binary:
        Y_binary = []
        for i in df[YVar]:
            if i >Y_mean:
                Y_binary.append(1)
            else:Y_binary.append(0)

        #print(f'The mean of Y is {np.mean(Y_binary)}')
        return Y_binary

    else:
        Y_sd = np.std(df[YVar])
        Y = []
        for i in df[YVar]:
            if i > Y_mean + 2 * Y_sd:
                Y.append(8)

            elif i > Y_mean + Y_sd:
                Y.append(7)

            elif i > Y_mean + 0.5 * Y_sd:
                Y.append(6)

            elif i > Y_mean:
                Y.append(5)

            elif i < (Y_mean - 2 * Y_sd):
                Y.append(1)

            elif i < (Y_mean - Y_sd):
                Y.append(2)

            elif i < (Y_mean - 0.5 * Y_sd):
                Y.append(3)

            else:
                Y.append(4)

        #print(f'The mean of Y is {np.mean(Y)}')
        return Y

def create_lags(df,timevar,variable,lags:list[int]):

    df = df.sort_values(by=timevar,axis=0,ascending=True)
    for i in lags:
        df[f'{variable}_{i}'] = df[variable].shift(i)

    return df

def create_moving_average(df,timevar,variable,periods):

    df = df.sort_values(by=timevar,axis=0,ascending=True)
    df = create_lags(df,timevar,variable,[1])

    df[f'{variable}_MA_{periods}_1'] = df[f'{variable}_1'].rolling(periods).mean()

    return df

def set_time_period(df,timevar,start,end):
    start = pd.to_datetime(start).tz_localize(pytz.utc)
    end = pd.to_datetime(end).tz_localize(pytz.utc)

    df[timevar] = pd.to_datetime(df[timevar])

    try:
        df[timevar] = df[timevar].dt.tz_localize(pytz.utc)
    except: None

    df = df.sort_values(by=timevar,axis=0,ascending=True)

    small_df = df[~(df[timevar] < start)]
    small_df = small_df[~(small_df[timevar] > end)]

    return small_df

def normalize(method,df,variable:str):

    if not method in ['MinMax','NormDist','DivMax']:
        method = 'NormDist'
        print('Value Error Warning: Invalid normalization method selected - using NormDist')

    if method == 'NormDIst':
        std = np.std(df[variable])
        mean = np.mean(df[variable])
        df[f'{variable}_norm'] = df[variable].map(lambda x: ((x-mean)/std) )

    elif method == 'MinMax':
        min = np.min(df[variable])
        max = np.max(df[variable])
        df[f'{variable}_norm'] = df[variable].map(lambda x: ((x-min )/ max-min))

    else: #method == 'DivMax'
        max = np.max(df[variable])
        df[f'{variable}_norm'] = df[variable].map(lambda x: x/max)



    return df

def trim_outliers(df,variable):

    df['varCat'] = Y_cat_format(df,variable,False)
    df = df[df['varCat'] < 6]
    df = df[df['varCat'] > 3]

    df.drop(columns = ['varCat'])

    return df

