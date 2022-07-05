---
output:
  pdf_document: default
  html_document: default
---
# Predicting the South African Bitcoin, Ethereum and Solana Markets using International Market Data

##1. Introduction

This paper examines the relationship between the international cryptocurrency market and the South African cryptocurrency market.
It is hypothesized that the South African market slightly lags the international market in terms of price movement. 
If this is the case and the relationship can be modelled it could lead to improvements in trading strategies for cryptocurrency traders.
It was found that...

This paper used data gathered from Kraken - an international cryptocurrency exchange - to represent the international cryptocurrency market 
and from VALR - a South African cryptocurrency exchange - to represent the South African market. In specific, the per minute pricing history
for the rand and dollar markets for Bitcoin, Ethereum and Solana were gathered from the two exchanges. Next the international markets were 
compared to the local markets to test for correlation and finally the relationships were modelled using multiple linear regressions, gradient 
boosting and neural network algorithms available from SciKit-Learn. 

The paper begins by describing the data collection process before detailing the methods of feature extraction used. 
Next, descriptive statistics are presented and their implications explained following which further data manipulation is undertaken 
in order to account for the idiosyncracies of the specific dataset. Next, the methods and results of the modelling process are presented and
finally the results are discussed in the conclusion.

##2. Data Collection and Cleaning
Data collection was done via email (and WeTransfer) on the VALR side and via download on the Kraken side. The VALR data was delivered as per minute pricing data describing the opening-price, high-price, low-price, close-price and volume per minute. This is known as the OHLC format. 
The VALR data was adjusted from the Pretoria, South Africa timezone to the UTC timezone in order to match the timezone of the Kraken data. No further cleaning was necessary for the VALR data.

Kraken provides trade history for the Ethereum-USD (ETH-USD) market and pricing history per minute for the Bitcoin-USD (BTC-USD) market in this [Google Drive](https://drive.google.com/drive/folders/1jI3mZvrPbInNAEaIOoMbWvFfgRDZ44TT "Kraken Historical Data Google Drive").
The ETH-USD trade history was converted into a per minute OHLC format by grouping the trades into minute long intervals and taking the opening trade price as the opening-price, the highest trade price as the high-price, the lowest trade price as the low-price, the final trade price as the closing-price
and the total volume of cryptocurrency traded as the volume. 

Because the aim of this project is to use the international market for Bitcoin and Ethereum to predict the South African market in order to augment algorithmic trading decisions - 
the per minute OHCL data for both markets was differenced by its first lag. In particular, the closing prices were differenced and converted to a percentage. 
One of the major benefits of differencing the data is that it centers it around zero. Please see Figures 1 and 2 for a comparison of the differenced and undifferenced 
BTC-USD closing prices over time. Further, differencing the data (by percentage) brings the scale of the two markets together. Because of the ZAR/USD exchange rate (around R15 per USD), the ZAR-BTC and ZAR-ETH markets have nominal values around 15 times higher than the USD markets.
This is problematic if both ZAR lags and USD lags are included in the variables input into models that work with node weightings because it can bias the model towards variables with bigger scales (reference here). This problem is mitigated by the scaling inherent in differencing by percentage.
After differencing, the first 5 lags of the closing prices of both the ZAR and USD markets were taken. Similarly, a 100 period moving average and a 5 period moving average were taken for both markets. Note that the moving averages are also lagged by one period in order to avoid any data contamination.

Finally, the differenced ZAR-BTC and USD-BTC variables were divided into 8 categories dependent on the number of standard deviations that a sample lies away from the mean. Please see Equation 1 for details.

![Figure 1](/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/BTC_ZAR_vs_time.png)  
*Figure 1: BTC-ZAR over Time*

![Figure 2](/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/BTC_ZAR_Diff_vs_time.png)  
*Figure 2: Differenced BTC-ZAR over Time*  

$$ Ycat=   \left\{
\begin{array}{ll}
      1 & where & \mu-2\sigma>Y \\
      2 & where & \mu-\sigma>Y>\mu-2\sigma \\
      3 & where & \mu-0,5\sigma>Y>\mu-\sigma \\
      4 & where & \mu>Y>\mu-0,5\sigma \\
      5 & where & \mu<Y<\mu+0,5\sigma \\
      6 & where & \mu+0,5\sigma<Y<\mu+\sigma \\
      7 & where & \mu+\sigma<Y<\mu+2\sigma \\
      8 & where & \mu+2\sigma<Y \\
\end{array} 
\right.  
$$
*Equation 1: Piecewise categorization of variables*

##Initial analysis
Single linear (OLS) regression was used to test the hypothesis in the most basic way and yielded a positive result. Regressing the un-lagged BTC-ZAR variable on the un-lagged BTC-USD variable reveals a strong positive relationship between the two - please see Figure 3.
Next, regressing the un-lagged BTC-ZAR variable on the  first lag of the BTC-USD market yields a weaker but comparable positive relationship - please see Figure 4. However, regressing the un-lagged BTC-USD variable on the first lag of the BTC-USD market reveals a very weak positive relationship
- please see Figure 5. In fact, even the second lag of the BTC-USD variable is a better predictor of the un-lagged BTC-ZAR market than the first lag of the BTC-USD variable is for the BTC-USD market. Please see Table 1 for coefficients and R-Squared values. While it is clear that any linear relationship 
between lags of the BTC-USD market and the un-lagged BTC-ZAR market is weak these results do suggest that the USD market leads the South African market at least in some way.

|Regression|Coefficient|R-Squared|
|----------|-----------|---------|
|BTC-ZARDiff on BTC-USDDiff| 0,5 |0,17|
|BTC-ZARDiff on BTC-USDDiff_1|0,26|0,05|
|BTC-USDDiff on BTC-USDDiff_1|0,03|0,0008|
|BTC-ZARDiff on BTC-USDDiff_2|0,08|0,004|

![Figure 3](/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/Scatter_ZAR_vs_USD.png)
*Figure 3: Unlagged BTC-ZAR vs unlagged BTC-USD*  
*R-squared: 0.17  , Coefficient:0.5*

![Figure 4](/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/Scatter_ZAR_vs_USD_1.png)  
*Figure 4: First lag of BTC-ZAR vs unlagged BTC-USD*  
*R-squared: 0.05  , Coefficient:0.27*

![Figure 5](/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/Scatter_USD_vs_USD_1.png)  
*Figure 5: First lag of BTC-USD vs unlagged BTC-USD*  
*R-squared: 0.0008  , Coefficient:0.03*

Following these results, machine learning models were tested. The variables selected for 







<!-- Images -->

![Figure 3](/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/HMap_ZAR_vs_USD_1.png)
![Figure 4](/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/Images/HMap_USD_vs_USD_1.png)


<-- Tables -->

|Var1|Var2|
|----|----|
|x1  | x2 |
|x1  | x2 |



