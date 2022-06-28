import pandas as pd
import numpy as np
import os
import datetime

def trades_to_OHLC_per_minute(df,startDate:str,endDate:str):
    df.columns = ['Timestamp', 'Price', 'Volume']

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

    df = df[~(df['Timestamp'] < startDate)]
    df = df[~(df['Timestamp'] > endDate)]


    df['Timestamp'] = df['Timestamp'].dt.floor('Min')

    uniqueMinutes = pd.Series(df['Timestamp'].unique())

    openList = []
    highList = []
    lowList = []
    closeList = []
    volumeList = []
    tradesList = []

    for i in uniqueMinutes:
        currentDf = df[(df['Timestamp'] == i)]
        currentDf.reset_index(drop=True, inplace=True)

        openList.append(currentDf.iloc[0]['Price'])
        highList.append(max(currentDf['Price']))
        lowList.append(min(currentDf['Price']))
        closeList.append(currentDf.iloc[-1]['Price'])
        volumeList.append(np.sum(currentDf['Volume']))
        tradesList.append(len(currentDf))

    OHLC_dict = {'Timestamp':list(uniqueMinutes),'Open':openList,'High':highList,'Low':lowList,'Close':closeList,
            'Volume':volumeList,'Trades':tradesList}

    OHLC_df = pd.DataFrame(OHLC_dict)

    return OHLC_df

def OHLC_snipper(df,startDate:str,endDate:str):
    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades']
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df[~(df['Timestamp'] < startDate)]
    df = df[~(df['Timestamp'] > endDate)]


    return df

filePath = '/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken'
startDate = '2021-01-01'
endDate = '2022-06-28'


for i in os.listdir(filePath):
    if not (i.startswith('.') or i.startswith('OHLC')):

        print(i)
        df = pd.read_csv(f'{filePath}/{i}')

        if len(df.columns) == 3:
            OHLC = trades_to_OHLC_per_minute(df,startDate,endDate)
            OHLC.to_csv(f'{filePath}/OHLC_{i}')


        elif len(df.columns) == 7:
            OHLC = OHLC_snipper(df,startDate,endDate)
            OHLC.to_csv(f'{filePath}/OHLC_{i}')
        else:
            print(f'{i} has an invalid data setup')


