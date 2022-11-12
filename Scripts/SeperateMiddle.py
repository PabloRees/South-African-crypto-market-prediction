import pandas as pd
from Descriptive_Statistics_Functions import setup_and_graph_fiat, plotOverTime, categorical_heat_map_from_conf_mat
from Std_fin_ts_data_setup import mark_middle, mark_middle2, up_down
from Shrinkage_methods import Shrinkage_Methods
from ML_Package import runML_tests
import numpy as np
from matplotlib import pyplot as plt
import pickle

BTCZAR = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/VALR/BTCZAR.csv')
BTCUSD = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_XBTUSD_1.csv')

imageFolderPath = '/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images'

startDate, endDate = '2020-01-01', '2022-04-01'

BTCFiat = setup_and_graph_fiat(BTCUSD,BTCZAR,startDate,endDate,diffSavePath=f'{imageFolderPath}/BTC_ZAR_Diff_vs_time.png'
                               ,unDiffSavePath=f'{imageFolderPath}/BTC_ZAR_vs_time.png')

BTCFiat = mark_middle(BTCFiat,'ZARDiff',[-0.1,0.1],'Timestamp')
BTCFiat = mark_middle2(BTCFiat,'ZARDiff',[-0.1,0.1],'Timestamp')
BTCFiat = up_down(BTCFiat,'ZARDiff','Timestamp')


ax = BTCFiat['Updown_tag'].plot.hist(bins=2, alpha=0.9)
ax.plot()
plt.show()

print('\nFull set\n')
print(BTCFiat['Updown_tag'].value_counts())
for i in BTCFiat['Updown_tag'].value_counts():
    print(f'{i/len(BTCFiat["Updown_tag"])}')

for i in BTCFiat.columns:
    print(i)

volVars = ['USDVolume_1','USDVolume_MA_5_1','USDVolume_MA_50_1','USDVolume_MA_100_1','USDVolume_MA_1000_1','ZARDiff_2','ZARVolume_1','ZARVolume_MA_5_1','ZARVolume_MA_50_1','ZARVolume_MA_100_1','ZARVolume_MA_1000_1']

refinedVolVars = ['ZARDiff_1','USDDiff_1','USDVolume_MA_100_1','USDVolume_MA_1000_1','ZARVolume_1','ZARVolume_MA_5_1','ZARVolume_MA_50_1','ZARVolume_MA_100_1','ZARVolume_MA_1000_1']

eightVars = ['USDVolume_MA_5_1','ZARDiff_2','ZARDiff_MA_100_1','ZARDiff_1','USDDiff_1','USDDiff_2','ZARVolume_1','ZARVolume_MA_5_1']

non_binary = ['ZARDiff_1','ZARDiff_2','USDDiff_1','USDDiff_2',
              'ZARVolume_1','ZARVolume_MA_100_1','NN_Section_pred',
              'NN2_Section_pred','GB_Section_pred']


ZARDiff = ['ZARDiff']

SectionTag = ['Section_tag']

def ML_2(df,clf_algo,startDate,XVars, YVar):
    BTCtrainScores, BTCtestScores, _ = runML_tests(full_df=df, XVars=XVars, YVar=YVar,
                                                   remove_duplicate_dates='False',
                                                   crossVals=5, scoring='accuracy', clf_type=clf_algo,
                                                   ML_type='TS_Classifier',
                                                   startDate=startDate, return_prediction=False, binary=False, keep_current_cat=True)

    print(f'Train accuracy:{np.mean(BTCtrainScores.accuracy)}\n'
          f'Train precision:{np.mean(BTCtrainScores.precision)}\n'
          f'Train recall:{np.mean(BTCtrainScores.recall)}')

    print(f'Test accuracy:{BTCtestScores.accuracy[4]}\n'
          f'Test precision:{BTCtestScores.precision[4]}\n'
          f'Test recall:{BTCtestScores.recall[4]}\n\n')

# load it again
with open('GB.pkl', 'rb') as fid:
    GB_clf = pickle.load(fid)

with open('NN.pkl', 'rb') as fid:
    NN_clf = pickle.load(fid)

with open('NN2.pkl', 'rb') as fid:
    NN2_clf = pickle.load(fid)

X_folds1 = np.array(BTCFiat[refinedVolVars])

NN_pred = NN_clf.predict(X_folds1)
GB_pred = GB_clf.predict(X_folds1)

BTCFiat['NN_Section_pred'] = NN_pred
BTCFiat['GB_Section_pred'] = GB_pred
AugmentedVars = refinedVolVars + ['NN_Section_pred','GB_Section_pred']

X_folds2 = np.array(BTCFiat[AugmentedVars])
NN2_pred = NN2_clf.predict(X_folds2)
BTCFiat['NN2_Section_pred'] = NN2_pred

AugmentedVars = refinedVolVars + ['NN_Section_pred','NN2_Section_pred','GB_Section_pred']

#Testing for ability to predict middle vs non middle (i.e. Binary)
#Shrinkage = Shrinkage_Methods(BTCFiat,AugmentedVars ,'Section_tag', 10 )
#Shrinkage.Elastic_Gridsearch(0.1,figSavePath=f'{imageFolderPath}/augmentedMidPredShrink.jpeg',show_coefficients=True,minAlpha=0,maxAlpha=10)

#print('##RefinedVol Neural Network Tests\n___________________________________________________________________________')
#ML_2(BTCFiat,'clf_NN',startDate,AugmentedVars,'Section_tag')

#print('##RefinedVol Gradient Boosting Tests\n___________________________________________________________________________')
#ML_2(BTCFiat,'clf_GradientBoosting',startDate,AugmentedVars,'Section_tag')

#Testing for ability to predict 4 categories around the middle (i.e. non-binary)
print('\n\nNon-binary tests ____________________________________________________________')
Shrinkage = Shrinkage_Methods(BTCFiat,non_binary ,'Updown_tag', 10 )
Shrinkage.Elastic_Gridsearch(0.1,figSavePath=f'{imageFolderPath}/augmentedMidPredShrink.jpeg',show_coefficients=True,minAlpha=0,maxAlpha=10)

print('##Non-binary Neural Network Tests\n___________________________________________________________________________')
ML_2(BTCFiat,'clf_NN',startDate,non_binary,'Updown_tag')

print('##Non-binary Gradient Boosting Tests\n___________________________________________________________________________')
ML_2(BTCFiat,'clf_GradientBoosting',startDate,non_binary,'Updown_tag')