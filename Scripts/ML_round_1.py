from Descriptive_Statistics_Functions import setup_and_graph_fiat
import pandas as pd
from ML_Package import runML_tests
import numpy as np
from matplotlib import pyplot as plt

BTCZAR = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/VALR/BTCZAR.csv')
BTCUSD = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_XBTUSD_1.csv')

imageFolderPath = '/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images'

startDate, endDate = '2020-01-01', '2022-04-01'

BTCFiat = setup_and_graph_fiat(BTCUSD,BTCZAR,startDate,endDate,diffSavePath=f'{imageFolderPath}/BTC_ZAR_Diff_vs_time.png'
                               ,unDiffSavePath=f'{imageFolderPath}/BTC_ZAR_vs_time.png')

ax = BTCFiat['ZARDiff'].plot.hist(bins=60, alpha=0.9)
ax.plot()
plt.title('Histogram of continuous ZAR-BTC')
plt.xlabel('Differenced ZAR-BTC')
plt.savefig('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/ZARDiff_cont_hist.png')
plt.show()

print(BTCFiat['ZARDiff'].describe())

ax2 = BTCFiat['ZARDiff_cat'].plot.hist(bins=8, alpha=0.9)
ax2.plot()
plt.title('Histogram of categorical ZAR-BTC')
plt.xlabel('Categories of standard deviations\nfrom the mean of ZAR-BTC')
plt.savefig('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/ZARDiff_cat_hist.png')
plt.show()


print(BTCFiat['ZARDiff_cat'].value_counts())
for i in BTCFiat['ZARDiff_cat'].value_counts():
    print(f'{i/len(BTCFiat["ZARDiff_cat"])}')


eightVars = ['USDVolume_MA_5_1','ZARDiff_2','ZARDiff_MA_100_1','ZARDiff_1','USDDiff_1','USDDiff_2','ZARVolume_1','ZARVolume_MA_5_1']

fiveVars = ['ZARDiff_2','ZARDiff_1','USDDiff_1','USDDiff_2','ZARVolume_1']


def ML_1(clf_algo,reg_algo,startDate,XVars, YVar):
    BTCtrainScores, BTCtestScores, _ = runML_tests(full_df=BTCFiat, XVars=XVars, YVar=YVar,
                                                   remove_duplicate_dates='False',
                                                   crossVals=5, scoring='accuracy', clf_type=clf_algo,
                                                   ML_type='TS_Classifier',
                                                   startDate=startDate, return_prediction=False, binary=True)

    print(f'BTC Train accuracy:{np.mean(BTCtrainScores.accuracy)}\n'
          f'BTC Train precision:{np.mean(BTCtrainScores.precision)}\n'
          f'BTC Train recall:{np.mean(BTCtrainScores.recall)}')

    print(f'BTC Test accuracy:{BTCtestScores.accuracy[4]}\n'
          f'BTC Test precision:{BTCtestScores.precision[4]}\n'
          f'BTC Test recall:{BTCtestScores.recall[4]}\n\n')

    BTCRegTrain, BTCRegTest, _ = runML_tests(full_df=BTCFiat, XVars=XVars, YVar=YVar,
                                             remove_duplicate_dates='False',
                                             crossVals=5, scoring='accuracy', reg_type=reg_algo, ML_type='TS_Regressor',
                                             startDate=startDate, return_prediction=False, binary=False)

    print(f'BTC ZARDiff std error: {np.std(BTCFiat["ZARDiff"])}')
    print(f'BTC Train MSE:{np.mean(BTCRegTrain.MSE)}\n'
          f'BTC Train MAE:{np.mean(BTCRegTrain.MAE)}')

    print(f'BTC Test MSE:{BTCRegTest.MSE[4]}\n'
          f'BTC Test MAE:{BTCRegTest.MAE[4]}\n\n')




print('##FiveVar Regression Tests\n___________________________________________________________________________')
ML_1('clf_logreg','reg_MLR',startDate,fiveVars,'ZARDiff')

print('##FiveVar Gradient Boosting Tests\n___________________________________________________________________________')
ML_1('clf_GradientBoosting','reg_GradientBoosting',startDate,fiveVars,'ZARDiff')

print('##FiveVar Neural Network Tests\n___________________________________________________________________________')
ML_1('clf_NN','reg_NN',startDate,fiveVars,'ZARDiff')


print('##EightVar Regression Tests\n___________________________________________________________________________')
ML_1('clf_logreg','reg_MLR',startDate,eightVars,'ZARDiff')

print('##EightVar Gradient Boosting Tests\n___________________________________________________________________________')
ML_1('clf_GradientBoosting','reg_GradientBoosting',startDate,eightVars,'ZARDiff')

print('##EightVar Neural Network Tests\n___________________________________________________________________________')
ML_1('clf_NN','reg_NN',startDate,eightVars,'ZARDiff')