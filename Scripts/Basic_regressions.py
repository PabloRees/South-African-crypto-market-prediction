import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Std_fin_ts_data_setup import create_lags, create_moving_average, set_time_period, normalize\
    ,trim_outliers, Y_cat_format
from ML_Tests import runML_tests
import pytz

SOLZAR = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/VALR/SOLZAR.csv')
ETHZAR = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/VALR/ETHZAR.csv')
BTCZAR = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/VALR/BTCZAR.csv')

SOLUSD = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_SOLUSD.csv')
BTCUSD = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_XBTUSD_1.csv')
ETHUSD = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_ETHUSD.csv')

SOLBTC = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_SOLXBT.csv')
ETHBTC = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_ETHXBT.csv')

ax = (BTCZAR['Close']-BTCZAR['Open']).plot.hist(bins=60, alpha=0.9)
ax.plot()
plt.show()

def plotOverTime(df):
    df['Day'] = df['Timestamp'].map(lambda x: x.split(' ')[-1])
    df = df[df['Day'] == '00:00:00+00:00']
    plt.plot(df['Timestamp'], df['Open'])
    plt.title('ETHZAR over time')
    plt.xlabel('Date')
    plt.ylabel('Time')
    plt.show()


def setup_and_graph_fiat(USDdf,ZARdf,openDate,closeDate,normMethod):

    ZARdf = set_time_period(ZARdf, 'Timestamp', openDate, closeDate)
    ZARdf['ZARDiff'] = 100 * (ZARdf['Close'] - ZARdf['Open']) / ZARdf['Open']
    ZARdf = create_lags(ZARdf, 'Timestamp', 'ZARDiff', [1, 2, 3, 4, 5])
    ZARdf = create_moving_average(ZARdf, 'Timestamp', 'ZARDiff', 100)
    ZARdf['ZARVolume'] = ZARdf['Volume']
    ZARdf = create_moving_average(ZARdf, 'Timestamp', 'ZARVolume', 5)
    ZARdf = trim_outliers(ZARdf,'ZARDiff')
    ZARdf = normalize(normMethod,ZARdf, 'ZARDiff')
    ZARdf.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Quote_Volume', 'Market'], inplace=True)

    USDdf = set_time_period(USDdf, 'Timestamp', openDate, closeDate)
    USDdf['Timestamp'] = pd.to_datetime(USDdf['Timestamp'])
    USDdf['USDDiff'] = 100 * (USDdf['Close'] - USDdf['Open']) / USDdf['Open']
    USDdf = create_lags(USDdf, 'Timestamp', 'USDDiff', [1, 2, 3, 4, 5])
    USDdf = create_moving_average(USDdf, 'Timestamp', 'USDDiff', 100)
    USDdf['USDVolume'] = USDdf['Volume']
    USDdf = create_moving_average(USDdf, 'Timestamp', 'USDVolume', 5)
    USDdf.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Trades'], inplace=True)

    mixedDf = USDdf.merge(ZARdf, on='Timestamp')
    mixedDf = mixedDf.dropna(how='any')


    print(mixedDf['ZARDiff_norm'].describe())
    ax = mixedDf['ZARDiff_norm'].plot.hist(bins=60, alpha=0.9)
    ax.plot()
    plt.show()

    mixedDf['ZARDiff_norm_Binary'] = Y_cat_format(mixedDf, 'ZARDiff_norm', False)
    ax2 = mixedDf['ZARDiff_norm_Binary'].plot.hist(bins=8, alpha=0.9)
    ax2.plot()
    plt.show()

    return mixedDf

def setup_and_graph_crypto(SOLdf,ETHdf,openDate,closeDate,normMethod):

    SOLdf = set_time_period(SOLdf, 'Timestamp', openDate, closeDate) #snips the dataset at the open and close time
    SOLdf['SOLDiff'] = 100 * (SOLdf['Close'] - SOLdf['Open']) / SOLdf['Open']
    SOLdf = create_lags(SOLdf, 'Timestamp', 'SOLDiff', [1, 2, 3, 4, 5])
    SOLdf = create_moving_average(SOLdf, 'Timestamp', 'SOLDiff', 100)
    SOLdf['SOLVolume'] = SOLdf['Volume']
    SOLdf = create_moving_average(SOLdf, 'Timestamp', 'SOLVolume', 5)
    SOLdf = normalize(normMethod,SOLdf, 'SOLDiff')
    SOLdf.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Trades'], inplace=True)

    ETHdf = set_time_period(ETHdf, 'Timestamp', openDate, closeDate)
    ETHdf['Timestamp'] = pd.to_datetime(ETHdf['Timestamp'])
    ETHdf['ETHDiff'] = 100 * (ETHdf['Close'] - ETHdf['Open']) / ETHdf['Open']
    ETHdf = create_lags(ETHdf, 'Timestamp', 'ETHDiff', [1, 2, 3, 4, 5])
    ETHdf = create_moving_average(ETHdf, 'Timestamp', 'ETHDiff', 100)
    ETHdf['ETHVolume'] = ETHdf['Volume']
    ETHdf = create_moving_average(ETHdf, 'Timestamp', 'ETHVolume', 5)
    ETHdf.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Trades'], inplace=True)

    mixedDf = SOLdf.merge(ETHdf, on='Timestamp')
    mixedDf.dropna(inplace=True)
    mixedDf = trim_outliers(mixedDf,'ZARDiff')


    print(mixedDf['SOLDiff_norm'].describe())
    ax = mixedDf['SOLDiff_norm'].plot.hist(bins=60, alpha=0.9)
    ax.plot()
    plt.show()

    mixedDf['SOLDiff_norm_Binary'] = Y_cat_format(mixedDf, 'SOLDiff_norm', False)
    ax2 = mixedDf['SOLDiff_norm_Binary'].plot.hist(bins=8, alpha=0.9)
    ax2.plot()
    plt.show()

    return mixedDf

BTCFiat = setup_and_graph_fiat(BTCUSD,BTCZAR,'2020-01-01','2022-04-01','MinMax')
ETH = setup_and_graph_fiat(ETHUSD,ETHZAR,'2020-01-01','2022-04-01','MinMax')
SOL = setup_and_graph_fiat(SOLUSD,SOLZAR,'2020-01-01','2022-04-01','MinMax')


exit()

reg_types = ['reg_SGD', 'reg_NN', 'reg_MLR','reg_GradientBoosting']
X = ['USDDiff_1','USDDiff_2','USDDiff_3','ZARDiff_1','ZARDiff_MA_100_1'] #This set gets around 80% accuracy on the test set but precisopn around 62%

#X = ['USDDiff_1','USDDiff_2','USDDiff_3','ZARDiff_1','USDDiff_MA_5_1']

print(BTCFiat.columns)
#Need to: Regress local BTC changes on international BTC changes

trainScores, testScores, valScores = runML_tests(full_df=BTCFiat, XVars=X, YVar='ZARDiff' ,
                                    remove_duplicate_dates='False',
                                    crossVals=5, scoring='accuracy', clf_type='clf_NN', ML_type='CS_Classifier',
                                    startDate='2020-01-01',return_prediction=False,binary=True)


print(f'Train accuracy:{trainScores.accuracy}\n'
      f'Train precision:{trainScores.precision}\n'
      f'Train recall:{trainScores.recall}')

print(f'Test accuracy:{testScores.accuracy}\n'
      f'Test precision:{testScores.precision}\n'
      f'Test recall:{testScores.recall}')
