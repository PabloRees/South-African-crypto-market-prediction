import pandas as pd
import numpy as np
from Std_fin_ts_data_setup import create_lags, create_moving_average, set_time_period\
    ,trim_outliers, Y_cat_format, plotOverTime, get_difference, trim_middle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from Shrinkage_methods import Shrinkage_Methods

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

def reg_and_scatter_plot(df,Xvar,Yvar,savePath):

    X = df[Xvar].to_numpy().reshape(-1, 1)
    Y = df[Yvar].to_numpy().reshape(-1, 1)

    reg = LinearRegression(fit_intercept=False, n_jobs=-1)
    reg.fit(X, Y)

    print(f'Regressing: {Yvar} on {Xvar}\n'
          f'R-squared : {reg.score(X, Y)} , Coefficient: {reg.coef_}\n')

    plt.plot(X, reg.predict(X), color='r')

    plt.scatter(df[Xvar], df[Yvar],cmap='cool')
    plt.xlabel(Xvar)
    plt.ylabel(Yvar)

    plt.savefig(savePath)

    plt.show()

def categorical_heat_map(mixedDf,var1,var2,savepath):

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
    plt.savefig(savepath)
    plt.show()


    return plt

def setup_and_graph_fiat(USDdf,ZARdf,openDate,closeDate,diffSavePath,unDiffSavePath):

    ZARdf = set_time_period(ZARdf, 'Timestamp', openDate, closeDate)

    ZARdf = get_difference(ZARdf,'Timestamp','Close','ZAR')
    ZARdf['ZARVolume'] = ZARdf['Volume']


    ZAR_vs_time = plotOverTime(ZARdf.tail(263520),'Close')
    ZAR_vs_time.savefig(unDiffSavePath)
    ZAR_vs_time.show()

    ZARDiff_vs_time = plotOverTime(ZARdf.tail(263520),'ZARDiff')
    ZARDiff_vs_time.savefig(diffSavePath)
    ZARDiff_vs_time.show()

    ZARdf = create_moving_average(ZARdf, 'Timestamp', 'ZARDiff', 100)
    ZARdf = create_lags(ZARdf, 'Timestamp', 'ZARDiff', [1, 2, 3, 4, 5])

    ZARdf = create_moving_average(ZARdf, 'Timestamp', 'ZARVolume', 5)
    ZARdf = create_moving_average(ZARdf, 'Timestamp', 'ZARDiff', 5)
    ZARdf.drop(columns=['Unnamed: 0','Open', 'High', 'Low', 'Close', 'Close_1', 'Volume','ZARVolume', 'Quote_Volume', 'Market'], inplace=True)


    USDdf = set_time_period(USDdf, 'Timestamp', openDate, closeDate)
    USDdf['Timestamp'] = pd.to_datetime(USDdf['Timestamp'])
    USDdf = get_difference(USDdf,'Timestamp','Close','USD')
    USDdf = create_lags(USDdf, 'Timestamp', 'USDDiff', [1, 2, 3, 4, 5])
    USDdf = create_moving_average(USDdf, 'Timestamp', 'USDDiff', 100)
    USDdf['USDVolume'] = USDdf['Volume']
    USDdf = create_moving_average(USDdf, 'Timestamp', 'USDVolume', 5)
    USDdf.drop(columns=['Unnamed: 0','Open', 'High', 'Low', 'Close', 'Close_1', 'Volume', 'Trades'], inplace=True)

    mixedDf = USDdf.merge(ZARdf, on='Timestamp')
    mixedDf = mixedDf.dropna(how='any')

    #categorize ZARDiff, USDDiff and USDDiff_1 - then plot USDDiffs against ZARDiffs

    mixedDf['ZARDiff_cat'] =  Y_cat_format(mixedDf, 'ZARDiff', False)
    mixedDf['USDDiff_cat'] =  Y_cat_format(mixedDf, 'USDDiff', False)
    mixedDf['USDDiff_1_cat'] =  Y_cat_format(mixedDf, 'USDDiff_1', False)
    mixedDf['USDDiff_2_cat'] =  Y_cat_format(mixedDf, 'USDDiff_2', False)


    ax = mixedDf['ZARDiff'].plot.hist(bins=60, alpha=0.9)
    ax.plot()
    plt.show()

    ax2 = mixedDf['ZARDiff_cat'].plot.hist(bins=8, alpha=0.9)
    ax2.plot()
    plt.show()

    return mixedDf

imageFolderPath = '/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images'

BTCFiat = setup_and_graph_fiat(BTCUSD,BTCZAR,'2020-01-01','2022-04-01',diffSavePath=f'{imageFolderPath}/BTC_ZAR_Diff_vs_time.png'
                               ,unDiffSavePath=f'{imageFolderPath}/BTC_ZAR_vs_time.png')

categorical_heat_map(BTCFiat,'ZARDiff_cat','USDDiff_cat',f'{imageFolderPath}/HMap_ZAR_vs_USD.png')
categorical_heat_map(BTCFiat,'ZARDiff_cat','USDDiff_1_cat',f'{imageFolderPath}/HMap_ZAR_vs_USD_1')
categorical_heat_map(BTCFiat,'USDDiff_cat','USDDiff_1_cat',f'{imageFolderPath}/HMap_USD_vs_USD_1')
categorical_heat_map(BTCFiat,'ZARDiff_cat','USDDiff_2_cat',f'{imageFolderPath}/HMap_ZAR_vs_USD_2')

reg_and_scatter_plot(BTCFiat,'USDDiff','ZARDiff',f'{imageFolderPath}/Scatter_ZAR_vs_USD')
reg_and_scatter_plot(BTCFiat,'USDDiff_1','ZARDiff',f'{imageFolderPath}/Scatter_ZAR_vs_USD_1')
reg_and_scatter_plot(BTCFiat,'USDDiff_1','USDDiff',f'{imageFolderPath}/Scatter_USD_vs_USD_1')
reg_and_scatter_plot(BTCFiat,'USDDiff_2','ZARDiff',f'{imageFolderPath}/Scatter_ZAR_vs_USD_2')

allVars = BTCFiat.columns

for i in allVars:
    print(i)

ElasticNetDf = BTCFiat.drop(columns=['USDDiff','Timestamp'])
ENVars = ElasticNetDf.columns

test1 = Shrinkage_Methods(ElasticNetDf,list(ENVars),'ZARDiff',5)
test1.Elastic_Gridsearch(0.05,show_coefficients=True,figSavePath='/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/ElsaticGrid_1')



test2Vars = ['USDVolume_MA_5_1','ZARDiff_2','ZARDiff_MA_100_1','ZARDiff_1','USDDiff_1','USDDiff_2','ZARVolume_1','ZARVolume_MA_5_1']
test2 = Shrinkage_Methods(ElasticNetDf,test2Vars,'ZARDiff',5)
test2.Elastic_Gridsearch(0.01,show_coefficients=True,figSavePath='/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/ElsaticGrid_2')






