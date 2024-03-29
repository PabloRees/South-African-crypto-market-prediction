import pandas as pd
import os
import pytz

filePath = '/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/VALR/OHCL_Data_BTC_ETH_SOL'
fileList = os.listdir(filePath)

df = pd.read_csv(f'{filePath}/{fileList[0]}')
fileList.pop(0)

for i in fileList:
    if not i.startswith('.'):
        new_df = pd.read_csv(f'{filePath}/{i}')
        df = pd.concat([df,new_df],axis=0)

df.columns = ['Timestamp', 'Market', 'Open', 'High', 'Low', 'Close', 'Volume', 'Quote_Volume']
df.drop_duplicates(inplace=True,ignore_index=True)

#setup timestamps to UTC time
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d, %H:%M')
SA_timezone = pytz.timezone('Africa/Johannesburg')
df['Timestamp'] = df['Timestamp'].dt.tz_localize(SA_timezone).dt.tz_convert(pytz.utc)

df = df[~(df['Timestamp'] < '2019-01-01')]


btcDf = df[(df['Market'] == 'BTCZAR')]
btcDf.reset_index(inplace=True)
btcDf.drop('index',axis=1,inplace=True)


ethDf = df[(df['Market'] == 'ETHZAR')]
ethDf.reset_index(inplace=True)
ethDf.drop('index',axis=1,inplace=True)


solDf = df[(df['Market'] == 'SOLZAR')]
solDf.reset_index(inplace=True)
solDf.drop('index',axis=1,inplace=True)


saveFilepath = '/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/VALR'

btcDf.to_csv(f'{saveFilepath}/BTCZAR.csv')
ethDf.to_csv(f'{saveFilepath}/ETHZAR.csv')
solDf.to_csv(f'{saveFilepath}/SOLZAR.csv')










