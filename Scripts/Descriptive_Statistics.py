import pandas as pd

from Shrinkage_methods import Shrinkage_Methods
from Descriptive_Statistics_Functions import  reg_and_scatter_plot, categorical_heat_map, setup_and_graph_fiat

SOLZAR = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/VALR/SOLZAR.csv')
ETHZAR = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/VALR/ETHZAR.csv')
BTCZAR = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/VALR/BTCZAR.csv')

SOLUSD = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_SOLUSD.csv')
BTCUSD = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_XBTUSD_1.csv')
ETHUSD = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_ETHUSD.csv')

SOLBTC = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_SOLXBT.csv')
ETHBTC = pd.read_csv('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Data/Kraken/OHLC_ETHXBT.csv')


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






