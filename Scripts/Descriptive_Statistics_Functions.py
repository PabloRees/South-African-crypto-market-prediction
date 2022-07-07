import numpy as np
from Std_fin_ts_data_setup import create_lags, create_moving_average, set_time_period\
    ,trim_outliers, Y_cat_format, plotOverTime, get_difference, trim_middle, cat_format_by_difference
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
import pandas as pd

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

    var1_labels = list(mixedDf[var1].unique())
    var2_labels = list(mixedDf[var1].unique())

    var1_labels.sort()
    var2_labels.sort()

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

def categorical_heat_map_from_conf_mat(confMat,savepath):

    print((np.size(confMat)))
    print(np.sqrt(np.size(confMat)))
    print(range(np.sqrt(np.size(confMat)).astype(np.int64)))
    var1_labels = range(np.sqrt(np.size(confMat)).astype(np.int64))
    var2_labels = var1_labels

    print(var1_labels)

    confMat = 100*(confMat/np.sum(confMat))
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

    ax.set_title(f"Percentages of corresponding categories\n between pred and actual")
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

    mixedDf['ZARDiff_cat_by_diff'] =  cat_format_by_difference(mixedDf, 'ZARDiff', 0.1)
    mixedDf['USDDiff_cat_by_diff_1'] =  cat_format_by_difference(mixedDf, 'USDDiff_1', 0.1)

    mixedDf['USDDiff_cat'] =  Y_cat_format(mixedDf, 'USDDiff', False)
    mixedDf['USDDiff_1_cat'] =  Y_cat_format(mixedDf, 'USDDiff_1', False)
    mixedDf['USDDiff_2_cat'] =  Y_cat_format(mixedDf, 'USDDiff_2', False)

    categorical_heat_map(mixedDf, 'ZARDiff_cat_by_diff', 'USDDiff_cat_by_diff_1', '/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/cat_by_diff_USD_1_vs_ZAR.png')

    ax = mixedDf['ZARDiff'].plot.hist(bins=60, alpha=0.9)
    ax.plot()
    plt.show()

    ax2 = mixedDf['ZARDiff_cat'].plot.hist(bins=8, alpha=0.9)
    ax2.plot()
    plt.show()

    return mixedDf

