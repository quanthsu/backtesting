import numpy as np
from scipy import stats
import configparser
import pandas as pd
from datetime import datetime, timedelta

config = configparser.ConfigParser()
config.read('strategy_config.ini')

strategy_path = config['base']['path']
report_path = config['base']['report_path']
report_plot_path = config['base']['report_plot_path']
report_plot_portfolio_path = config['base']['report_plot_portfolio_path']
trading_day_date_path = config['base']['trading_day_date_path']
file_type = config['base']['file_type']
year_days = int(config['base']['year_days'])
transaction_cost = float(config['base']['transaction_cost'])
risk_free_rate = float(config['base']['risk_free_rate'])
total_asset = float(config['allocation']['total_asset'])
twa02_path = config['base']['twa02_path']
twc02_path = config['base']['twc02_path']

Rf = 0 # pow(1.01, 1 / 365) - 1

def get_MDD(profit_series):
    return round((profit_series.cumsum().cummax() - profit_series.cumsum()).max(), 0)

def get_sharpe_ratio(return_series, year_days, risk_free_rate):
    mean = return_series.mean() * year_days - risk_free_rate
    sigma = return_series.std() * np.sqrt(year_days)
    return mean / sigma
    
def get_sortino_ratio(series, year_days, risk_free_rate):
    mean = series.mean() * year_days - risk_free_rate
    std_neg = series[series<0].std()*np.sqrt(year_days)
    return mean/std_neg
    
def get_volatility(return_series, year_days):
    return return_series.std() * np.sqrt(year_days)

def get_index_df(index_path, index_name):
    def preprocess(daily_df):
        stock_ids = list(daily_df['股票代號'].astype('str'))
        date_indexes = daily_df.iloc[:,2:].columns
        date_indexes = [datetime.strptime(date_index[:8], "%Y%m%d") for date_index in date_indexes]
        daily_df = daily_df.T.iloc[2:, :]
        daily_df.columns = stock_ids
        daily_df.index = date_indexes
        return daily_df

    twa_close_df = pd.read_excel(index_path, sheet_name='close', header=4).dropna()
    twa_close_df = preprocess(twa_close_df)
    twa_close_df[f'{index_name}_profit'] = (twa_close_df[index_name] / twa_close_df[index_name].shift(1) - 1).astype(float)
    twa_close_df['Date'] = twa_close_df.index
    return twa_close_df 

def get_occupy_ratio(hm):
    hm_line = hm.replace(0, np.nan).dropna().quantile(0.7)
    hm_adj = (hm_line - hm)
    hm_adj = hm_adj[hm_adj > 0]

    return  1 - hm_adj.sum() / (hm_line * len(hm_adj))

def evaluate_portfolio(daily_report, portfolio_cols):
    portfolio_df = daily_report.copy()
    portfolio_profit_cols = [col + '_profit' for col in portfolio_cols] 
    portfolio_holding_money_cols = [col + '_holding_money' for col in portfolio_cols] 
    portfolio_df['profit'] = portfolio_df[portfolio_profit_cols].sum(axis=1)
    portfolio_df['holding_money'] = portfolio_df[portfolio_holding_money_cols].sum(axis=1)

    net_profit = round(portfolio_df['profit'].sum(), 0)
    max_holding_money = portfolio_df['holding_money'].max()
    mdd_money = round((portfolio_df['profit'].cumsum().cummax() - portfolio_df['profit'].cumsum()).max(), 0)

    asset = portfolio_df['holding_money'].quantile(0.95)
    mdd_ratio = mdd_money / asset
    
    occupy_ratio = get_occupy_ratio(portfolio_df['holding_money'])

    portfolio_df['profit_pct'] = (portfolio_df['profit'].cumsum() + asset).pct_change()

    sharpe_ratio = round(get_sharpe_ratio(portfolio_df['profit'] / asset, year_days, risk_free_rate), 2)
    sortino_ratio = round(get_sortino_ratio(portfolio_df['profit'] / asset, year_days, risk_free_rate), 2)
    volatility = round(get_volatility(portfolio_df['profit'] / asset, year_days) * 100, 2) 

    twa02_df = get_index_df(twa02_path, 'TWA02')
    #portfolio_df = pd.merge(portfolio_df[['Date', 'profit']], twa02_df[['Date', 'TWA02_profit']], on=['Date'], how='inner')
    portfolio_df = pd.merge(portfolio_df[['Date', 'profit_pct']], twa02_df[['Date', 'TWA02_profit']], on=['Date'], how='inner').dropna()

    #beta, alpha, r_value, p_value, std_err = stats.linregress(portfolio_df['profit_pct'], portfolio_df['TWA02_profit'])
    beta = np.cov(portfolio_df['profit_pct'], portfolio_df['TWA02_profit'])[0][1] / np.var(portfolio_df['TWA02_profit'])
    alpha = ((portfolio_df['profit_pct'] - Rf) - beta * (portfolio_df['TWA02_profit']) - Rf).mean()

    return round(net_profit/10000, 2), sharpe_ratio, sortino_ratio, round(mdd_money/10000, 2), round(mdd_ratio * 100, 2), round(max_holding_money/10000, 2), round(occupy_ratio * 100, 2), volatility, \
            round(beta, 2), round(alpha, 6)
