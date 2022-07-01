import pandas as pd
import numpy as np
from Std_fin_ts_data_setup import create_lags, create_moving_average, set_time_period, normalize\
    ,trim_outliers, Y_cat_format, plotOverTime, get_difference, trim_middle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

SOLZAR = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/VALR/SOLZAR.csv')
ETHZAR = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/VALR/ETHZAR.csv')
BTCZAR = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/VALR/BTCZAR.csv')

SOLUSD = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_SOLUSD.csv')
BTCUSD = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_XBTUSD_1.csv')
ETHUSD = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_ETHUSD.csv')

SOLBTC = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_SOLXBT.csv')
ETHBTC = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_ETHXBT.csv')

def get_cbar(data, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    return cbar

def categorical_heat_map(mixedDf,var1,var2):

    nrSamples = len(mixedDf)

    var1_labels = ["1", "2", "3", "4",
                  "5", "6", "7","8"]

    var2_labels = ["1", "2", "3", "4",
                  "5", "6", "7","8"]

    confMat = confusion_matrix(mixedDf[var1],mixedDf[var2])
    confMat = 100*(confMat/nrSamples)
    confMat = np.round(confMat,1)


    fig, ax = plt.subplots()
    heatplot = ax.imshow(confMat,cmap='cool')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(var2_labels)), labels=var2_labels)
    ax.set_yticks(np.arange(len(var1_labels)), labels=var1_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(var1_labels)):
        for j in range(len(var2_labels)):
            text = ax.text(j, i, confMat[i, j],
                           ha="center", va="center", color="w")

    cbar = get_cbar(confMat, ax=ax,
            cbar_kw={}, cbarlabel="Matching categories")

    ax.set_title(f"Percentages of corresponding categories\n between {var1} and {var2}")
    fig.tight_layout()
    plt.show()

def setup_and_graph_fiat(USDdf,ZARdf,openDate,closeDate,normMethod):

    ZARdf = set_time_period(ZARdf, 'Timestamp', openDate, closeDate)

    #ZARdf['ZARDiff'] = ZARdf['Close'] - ZARdf['Open']
    ZARdf = get_difference(ZARdf,'Timestamp','Close','ZAR')

    plotOverTime(ZARdf,'ZARDiff')

    ZARdf = create_moving_average(ZARdf, 'Timestamp', 'ZARDiff', 100)
    ZARdf['ZARVolume'] = ZARdf['Volume']
    ZARdf = create_moving_average(ZARdf, 'Timestamp', 'ZARVolume', 5)
    ZARdf.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Quote_Volume', 'Market'], inplace=True)

    ZARdf = normalize(normMethod,ZARdf, 'ZARDiff')



    USDdf = set_time_period(USDdf, 'Timestamp', openDate, closeDate)
    USDdf['Timestamp'] = pd.to_datetime(USDdf['Timestamp'])
    USDdf = get_difference(USDdf,'Timestamp','Close','USD')
    USDdf = create_lags(USDdf, 'Timestamp', 'USDDiff', [1, 2, 3, 4, 5])
    USDdf = create_moving_average(USDdf, 'Timestamp', 'USDDiff', 100)
    USDdf['USDVolume'] = USDdf['Volume']
    USDdf = create_moving_average(USDdf, 'Timestamp', 'USDVolume', 5)
    USDdf.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Trades'], inplace=True)

    mixedDf = USDdf.merge(ZARdf, on='Timestamp')
    mixedDf = mixedDf.dropna(how='any')

    mixedDf = normalize(normMethod,mixedDf, 'ZARDiff')
    mixedDf = normalize(normMethod,mixedDf, 'USDDiff')
    mixedDf = create_lags(mixedDf, 'Timestamp', 'USDDiff_norm', [1, 2])

    #categorize ZARDiff, USDDiff and USDDiff_1 - then plot USDDiffs against ZARDiffs

    mixedDf['ZARDiff_cat'] =  Y_cat_format(mixedDf, 'ZARDiff', False)
    mixedDf['USDDiff_cat'] =  Y_cat_format(mixedDf, 'USDDiff', False)
    mixedDf['USDDiff_1_cat'] =  Y_cat_format(mixedDf, 'USDDiff_1', False)
    mixedDf['USDDiff_2_cat'] =  Y_cat_format(mixedDf, 'USDDiff_2', False)

    ax = mixedDf['ZARDiff_norm'].plot.hist(bins=60, alpha=0.9)
    ax.plot()
    plt.show()

    mixedDf['ZARDiff_norm_non-Binary'] = Y_cat_format(mixedDf, 'ZARDiff_norm', False)
    ax2 = mixedDf['ZARDiff_norm_non-Binary'].plot.hist(bins=8, alpha=0.9)
    ax2.plot()
    plt.show()

    return mixedDf

BTCFiat = setup_and_graph_fiat(BTCUSD,BTCZAR,'2020-01-01','2022-04-01','NormDist')

categorical_heat_map(BTCFiat,'ZARDiff_cat','USDDiff_cat')
categorical_heat_map(BTCFiat,'ZARDiff_cat','USDDiff_1_cat')
categorical_heat_map(BTCFiat,'USDDiff_1_cat','USDDiff_2_cat')
categorical_heat_map(BTCFiat,'ZARDiff_cat','USDDiff_2_cat')