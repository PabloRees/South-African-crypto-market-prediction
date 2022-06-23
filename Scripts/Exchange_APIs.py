import requests
import pandas as pd
import time
import hashlib
import hmac
import datetime

def krakenDataCollector(pair):

    resp = requests.get(f'https://api.kraken.com/0/public/OHLC?pair={pair}')

    krakendf = pd.DataFrame(resp.json()['result'][pair])
    krakendf.columns = ['TimeStamp','Open','High','Low','Close','AdjClose','Volume','Trades']

    print(krakendf)

    print(krakendf.columns)

    return krakendf


def valrDataCollector(startTime,endTime,pair,API_Key):

    path = f'/v1/marketdata/{pair}/tradehistory?startTime={startTime}&endTime={endTime}'

    timestamp = int(time.time()*1000)

    signature = getValrSignature(api_key_secret=API_Key,timestamp=timestamp,verb='GET',path=path)

    url = f"https://api.valr.com{path}"

    payload = {}
    headers = {
        'X-VALR-API-KEY': API_Key,
        'X-VALR-SIGNATURE': signature,
        'X-VALR-TIMESTAMP': str(timestamp)
    }

    params = {

        'startTime' : startTime,
        'endTime' : endTime
    }

    response = requests.request("GET", url, headers=headers, data=payload,params=params)

    print(response.text)

def getValrSignature(api_key_secret, timestamp, verb, path, body=""):

    """Signs the request payload using the api key secret
    api_key_secret - the api key secret
    timestamp - the unix timestamp of this request e.g. int(time.time()*1000)
    verb - Http verb - GET, POST, PUT or DELETE
    path - path excluding host name, e.g. '/v1/withdraw
    body - http request body as a string, optional
    """
    payload = "{}{}{}{}".format(timestamp, verb.upper(), path, body)
    message = bytearray(payload, 'utf-8')
    signature = hmac.new(bytearray(api_key_secret, 'utf-8'), message, digestmod=hashlib.sha512).hexdigest()

    return signature


#Kraken parameters
krakenPair = 'XXBTZUSD' # pairs: 'XXBTZUSD',

#VALR parameters
valrPair = 'BTCZAR'
valrStartTime = pd.to_datetime('2021-01-01').isoformat()
valrEndTime = datetime.datetime.now().isoformat()

apiDf = pd.read_csv('Data/.API')

API_Key = apiDf.iloc[0]['API_Key']
API_Secret = apiDf.iloc[0]['API_Secret']


krakenDataCollector(krakenPair)

valrDataCollector(startTime=valrStartTime, endTime=valrEndTime,pair=valrPair,API_Key=API_Key)




