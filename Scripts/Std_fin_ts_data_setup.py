import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt

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


def cat_format_by_difference(df,YVar,difference):
    Y = []
    for i in df[YVar]:
        if i > difference:
            Y.append(4)

        elif i > 0:
            Y.append(3)

        elif i > -difference:
            Y.append(2)

        else: # i < difference
            Y.append(1)

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

    if method == 'NormDist':
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

def trim_outliers(df,variable,range:list[int]):

    df['varCat'] = Y_cat_format(df,variable,False)
    df = df[df['varCat'] < range[1]]
    df = df[df['varCat'] > range[0]]

    df.drop(columns = ['varCat'])

    return df

def trim_middle(df,variable,range:list[int],timevar:str):

    #df['varCat'] = Y_cat_format(df,variable,False)
    df1 = df[df[variable] < range[0]]
    df2 = df[df[variable] > range[1]]

    df = pd.concat([df1, df2], axis=0)
    df = df.sort_values(by=timevar,axis=0,ascending=True)

    #df.drop(columns = ['varCat'])

    return df

def mark_middle(df,variable,range:list[float],timevar:str):

    df_3 = df[df[variable] < range[1]]
    df3 = df_3[df_3[variable] > 0] #this should be between 0 and the upper range
    df3 = df3.assign(Section_tag=1)

    df4 = df[df[variable] > range[1]] #this should be above the upper range
    df4 = df4.assign(Section_tag=0)

    df_2 = df[df[variable] > range[0]]
    df2 = df_2[df_2[variable] < 0] #this should be between 0 and the lower range
    df2 = df2.assign(Section_tag=1)

    df1 = df[df[variable] < range[0] ] #this should be below the lower range
    df1 = df1.assign(Section_tag=0)

    df = pd.concat([df1, df2, df3, df4], axis=0)
    df = df.sort_values(by=timevar,axis=0,ascending=True)

    #df.drop(columns = ['varCat'])

    return df

def mark_middle2(df,variable,range:list[float],timevar:str):

    df_3 = df[df[variable] < range[1]]
    df3 = df_3[df_3[variable] > 0] #this should be between 0 and the upper range
    df3 = df3.assign(Section_tag2=1)

    df4 = df[df[variable] > range[1]] #this should be above the upper range
    df4 = df4.assign(Section_tag2=2)

    df_2 = df[df[variable] > range[0]]
    df2 = df_2[df_2[variable] < 0] #this should be between 0 and the lower range
    df2 = df2.assign(Section_tag2=-1)

    df1 = df[df[variable] < range[0] ] #this should be below the lower range
    df1 = df1.assign(Section_tag2=-2)

    df = pd.concat([df1, df2, df3, df4], axis=0)
    df = df.sort_values(by=timevar,axis=0,ascending=True)

    #df.drop(columns = ['varCat'])

    return df

def up_down(df,variable,timevar:str):

    df1 = df[df[variable] < 0 ] #this should be below 0
    df1 = df1.assign(Updown_tag=0)

    df2 = df[df[variable] >0 ] #this should be above 0
    df2 = df2.assign(Updown_tag=1)

    df = pd.concat([df1, df2], axis=0)
    df = df.sort_values(by=timevar,axis=0,ascending=True)

    return df


def plotOverTime(df,var):
    df['Day'] = df['Timestamp'].map(lambda x: str(x).split(' ')[-1])
    df = df[df['Day'] == '00:00:00+00:00']
    plt.plot(df['Timestamp'], df[var])
    plt.title(f'{var} over time')
    plt.xlabel('Date')
    plt.ylabel(f'{var}')
    return plt

def get_difference(df,timevar, variable, label):

    df = df.sort_values(by=timevar,axis=0,ascending=True)
    df = create_lags(df,timevar,variable,[1])
    df[f'{label}Diff'] = 100*(round(1000*df[variable]) - round(1000*df[f'{variable}_1']))/round(1000*df[f'{variable}_1'])

    return df