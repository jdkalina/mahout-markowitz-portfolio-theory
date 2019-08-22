import pandas as pd
import numpy as np

na_eq = pd.read_csv('SPX Price history.csv')


tickers = na_eq['Ticker'].unique()
for i in tickers:
    db = na_eq[na_eq['Ticker'] == i]
    rics = db['RIC'].unique()
    if len(rics) > 1:
    na_eq.drop(na_eq[na_eq['RIC'] == rics[1]].index, inplace = True)
    
    na_eq['Trade Date'] = pd.to_datetime(na_eq['Trade Date'])
    na_eq1 = na_eq.pivot_table(index = 'Trade Date', values = 'Universal Close Price', columns = 'Ticker')
    
    #not-annualized
    returns = np.log(na_eq1/na_eq1.shift(1))[1:].fillna(0)
    
    returns.to_csv(\"data.csv\")


num_ports = 10000
all_weights = np.zeros((num_ports, len(na_eq.columns))) #generates random portfolio weights
ret_arr = np.zeros(num_ports) #holds the portfolio returns we calculate
vol_arr = np.zeros(num_ports) #holds the volatilities that we calculate
sharpe_arr = np.zeros(num_ports) #holds the Sharpe Ratios that we calculate


for ind in range(num_ports):
    #need to rebalance so they equal 1
    weights = np.array(np.random.random(len(na_eq.columns))) # make this a matrix
    weights = weights/sum(weights) # this still
    all_weights[ind,:] = weights #skip this
    
    #expected return step
    ret_arr[ind] = np.sum( (na_eq_log.mean() * weights) * 252) # Weights Matrix %*% mean vector
    
    #expected volatility
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(na_eq_log.cov()*252, weights)))
    
    #Sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

