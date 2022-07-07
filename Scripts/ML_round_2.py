from Descriptive_Statistics_Functions import setup_and_graph_fiat, plotOverTime, categorical_heat_map_from_conf_mat
import pandas as pd
from ML_Package import runML_tests
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from Std_fin_ts_data_setup import trim_middle

BTCZAR = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/VALR/BTCZAR.csv')
BTCUSD = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_XBTUSD_1.csv')

imageFolderPath = '/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images'

startDate, endDate = '2020-01-01', '2022-04-01'

BTCFiat = setup_and_graph_fiat(BTCUSD,BTCZAR,startDate,endDate,diffSavePath=f'{imageFolderPath}/BTC_ZAR_Diff_vs_time.png'
                               ,unDiffSavePath=f'{imageFolderPath}/BTC_ZAR_vs_time.png')


ax2 = BTCFiat['ZARDiff_cat_by_diff'].plot.hist(bins=4, alpha=0.9)
ax2.plot()
plt.title('Histogram of categorical ZAR-BTC')
plt.xlabel('Categories of standard deviations\nfrom the mean of ZAR-BTC')
#plt.savefig('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/ZARDiff_cat_hist.png')
plt.show()

diff_cat = plotOverTime(BTCFiat, 'ZARDiff_cat_by_diff')
diff_cat.show()


eightVars = ['USDVolume_MA_5_1','ZARDiff_2','ZARDiff_MA_100_1','ZARDiff_1','USDDiff_1','USDDiff_2','ZARVolume_1','ZARVolume_MA_5_1']

fiveVars = ['ZARDiff_2','ZARDiff_1','USDDiff_1','USDDiff_2','ZARVolume_1']

fiveVarDf = BTCFiat[['Timestamp','ZARDiff_cat_by_diff','USDDiff_cat_by_diff_1','ZARDiff_2','ZARDiff_1','USDDiff_1','USDDiff_2','ZARVolume_1']]

print('\nFull set\n')
print(BTCFiat['ZARDiff_cat_by_diff'].value_counts())
for i in BTCFiat['ZARDiff_cat_by_diff'].value_counts():
    print(f'{i/len(BTCFiat["ZARDiff_cat_by_diff"])}')

noMidBTCFiatDf = trim_middle(BTCFiat, 'ZARDiff_cat_by_diff', [2, 3], 'Timestamp')

print('\nFull set\n')
print(noMidBTCFiatDf['ZARDiff_cat_by_diff'].value_counts())
for i in noMidBTCFiatDf['ZARDiff_cat_by_diff'].value_counts():
    print(f'{i/len(noMidBTCFiatDf["ZARDiff_cat_by_diff"])}')


def warmStart1():
    trainset, testset = train_test_split(fiveVarDf, test_size=0.2)
    testset, valset = train_test_split(testset, test_size=0.25)


    trainset_lessmid_1 = trim_middle(trainset,'ZARDiff_cat_by_diff',[2,2],'Timestamp')
    trainset_lessmid_2 = trim_middle(trainset,'ZARDiff_cat_by_diff',[2,3],'Timestamp')

    trainset_lessmid = pd.concat([trainset_lessmid_1, trainset_lessmid_1], axis=0)
    trainset_lessmid = pd.concat([trainset_lessmid, trainset_lessmid_2], axis=0)
    trainset_lessmid = pd.concat([trainset_lessmid, trainset_lessmid_2], axis=0)
    trainset_lessmid = pd.concat([trainset_lessmid, trainset_lessmid_2], axis=0)
    trainset_lessmid = pd.concat([trainset_lessmid, trainset_lessmid_2], axis=0)
    trainset_lessmid = pd.concat([trainset_lessmid, trainset], axis=0)
    trainset_lessmid = pd.concat([trainset_lessmid, trainset.head(360000)], axis=0)

    trainset_lessmid.sort_values(by='Timestamp')

    #_, trainset_lessmid = train_test_split(trainset_lessmid, test_size=0.25)





    #trainset_lessmid = pd.concat([trainset_lessmid, trainset.head(24000)], axis=0)
    #trainset_lessmid = pd.concat([trainset_lessmid, trainset.tail(24000)], axis=0)

    print('\nTrainset\n')
    print(trainset['ZARDiff_cat_by_diff'].value_counts())
    for i in trainset['ZARDiff_cat_by_diff'].value_counts():
        print(f'{i / len(trainset["ZARDiff_cat_by_diff"])}')

    print('\nJiggled trainset\n')
    print(trainset_lessmid['ZARDiff_cat_by_diff'].value_counts())
    for i in trainset_lessmid['ZARDiff_cat_by_diff'].value_counts():
        print(f'{i / len(trainset_lessmid["ZARDiff_cat_by_diff"])}')


    nn = MLPClassifier(hidden_layer_sizes=(10, 8, 4), alpha=0.01,
                        max_iter=10000, random_state=42, early_stopping=True, warm_start=True, learning_rate='constant',
                        n_iter_no_change=100)

    gb = GradientBoostingClassifier(random_state=42,learning_rate=0.01,
                                        min_samples_split=0.001,min_samples_leaf=0.1,max_depth=4)


    clf = gb


    clf.fit(trainset_lessmid[fiveVars], trainset_lessmid['ZARDiff_cat_by_diff'])
    #clf.fit(trainset[fiveVars], trainset['ZARDiff_cat_by_diff'])




    print('Training scores_________________\n')
    trainPred = clf.predict(trainset[fiveVars])
    trainConfMatSavePath = '/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/HMap_ML_2_test.png'
    print(pd.Series(trainPred).unique())
    trainConfmat = confusion_matrix(trainset['ZARDiff_cat_by_diff'], trainPred)
    print(trainConfmat)
    categorical_heat_map_from_conf_mat(trainConfmat, trainConfMatSavePath)

    print('Testing scores_________________\n')
    testPred = clf.predict(testset[fiveVars])
    print(pd.Series(testPred).unique())
    testConfmat = confusion_matrix(testset['ZARDiff_cat_by_diff'], testPred)
    print(testConfmat)
    testConfMatSavePath = '/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/HMap_ML_2_test.png'
    categorical_heat_map_from_conf_mat(testConfmat, testConfMatSavePath)

    exit()


def warmStart2():

    noMidFiveVarDf = trim_middle(fiveVarDf,'ZARDiff_cat_by_diff',[2,3],'Timestamp')

    trainset, testset = train_test_split(noMidFiveVarDf, test_size=0.2)
    testset, valset = train_test_split(testset, test_size=0.25)


    print('\nNo Mid Full Set \n')
    print(noMidFiveVarDf['ZARDiff_cat_by_diff'].value_counts())
    for i in noMidFiveVarDf['ZARDiff_cat_by_diff'].value_counts():
        print(f'{i / len(noMidFiveVarDf["ZARDiff_cat_by_diff"])}')

    print('\nNo Mid Trainset \n')
    print(trainset['ZARDiff_cat_by_diff'].value_counts())
    for i in trainset['ZARDiff_cat_by_diff'].value_counts():
        print(f'{i / len(trainset["ZARDiff_cat_by_diff"])}')

    print('\nNo Mid Testset \n')
    print(testset['ZARDiff_cat_by_diff'].value_counts())
    for i in testset['ZARDiff_cat_by_diff'].value_counts():
        print(f'{i / len(testset["ZARDiff_cat_by_diff"])}')




    nn = MLPClassifier(hidden_layer_sizes=(10, 8, 4), alpha=0.01,
                        max_iter=10000, random_state=42, early_stopping=True, warm_start=True, learning_rate='constant',
                        n_iter_no_change=100)

    gb = GradientBoostingClassifier(random_state=42,learning_rate=0.01,
                                        min_samples_split=0.001,min_samples_leaf=0.1,max_depth=4)


    clf = gb


    clf.fit(trainset[fiveVars], trainset['ZARDiff_cat_by_diff'])
    #clf.fit(trainset[fiveVars], trainset['ZARDiff_cat_by_diff'])




    print('Training scores_________________\n')
    trainPred = clf.predict(trainset[fiveVars])
    trainConfMatSavePath = '/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/HMap_ML_2_test.png'
    print(pd.Series(trainPred).unique())
    trainConfmat = confusion_matrix(trainset['ZARDiff_cat_by_diff'], trainPred)
    print(trainConfmat)
    categorical_heat_map_from_conf_mat(trainConfmat, trainConfMatSavePath)

    print('Testing scores_________________\n')
    testPred = clf.predict(testset[fiveVars])
    print(pd.Series(testPred).unique())
    testConfmat = confusion_matrix(testset['ZARDiff_cat_by_diff'], testPred)
    print(testConfmat)
    testConfMatSavePath = '/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/HMap_ML_2_test.png'
    categorical_heat_map_from_conf_mat(testConfmat, testConfMatSavePath)

    exit()


def ML_2(df,clf_algo,startDate,XVars, YVar):
    BTCtrainScores, BTCtestScores, _ = runML_tests(full_df=df, XVars=XVars, YVar=YVar,
                                                   remove_duplicate_dates='False',
                                                   crossVals=5, scoring='accuracy', clf_type=clf_algo,
                                                   ML_type='TS_Classifier',
                                                   startDate=startDate, return_prediction=False, binary=False)

    print(f'BTC Train accuracy:{np.mean(BTCtrainScores.accuracy)}\n'
          f'BTC Train precision:{np.mean(BTCtrainScores.precision)}\n'
          f'BTC Train recall:{np.mean(BTCtrainScores.recall)}')

    print(f'BTC Test accuracy:{BTCtestScores.accuracy[4]}\n'
          f'BTC Test precision:{BTCtestScores.precision[4]}\n'
          f'BTC Test recall:{BTCtestScores.recall[4]}\n\n')

print('##FiveVar Regression Tests\n___________________________________________________________________________')
ML_2(noMidBTCFiatDf,'clf_logreg',startDate,fiveVars,'ZARDiff')

print('##FiveVar Neural Network Tests\n___________________________________________________________________________')
ML_2(noMidBTCFiatDf,'clf_NN',startDate,fiveVars,'ZARDiff')

print('##FiveVar Gradient Boosting Tests\n___________________________________________________________________________')
ML_2(noMidBTCFiatDf,'clf_GradientBoosting',startDate,fiveVars,'ZARDiff')


print('##EightVar Regression Tests\n___________________________________________________________________________')
ML_2(noMidBTCFiatDf,'clf_logreg',startDate,eightVars,'ZARDiff')

print('##EightVar Neural Network Tests\n___________________________________________________________________________')
ML_2(noMidBTCFiatDf,'clf_NN',startDate,eightVars,'ZARDiff')

print('##EightVar Gradient Boosting Tests\n___________________________________________________________________________')
ML_2(noMidBTCFiatDf,'clf_GradientBoosting',startDate,eightVars,'ZARDiff')

