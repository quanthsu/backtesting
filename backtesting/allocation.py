
import numpy as np
import pandas as pd
import datetime
import glob
#from tqdm.auto import tqdm
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
#import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import deque

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF8')

import warnings
warnings.filterwarnings("ignore")

import configparser

config = configparser.ConfigParser()
config.read('strategy_config.ini')

strategy_path = config['base']['path']
report_path = config['base']['report_path']
report_plot_path = config['base']['report_plot_path']
trading_day_date_path = config['base']['trading_day_date_path']
file_type = config['base']['file_type']
img_path = config['base']['img_path']
twa02_path = config['base']['twa02_path']
strategy_report_paths = glob.glob(strategy_path + '*' + file_type)
intraday_start_day = datetime.strptime(config['base']['intraday_start_day'], '%Y-%m-%d')
year_days = int(config['base']['year_days'])
transaction_cost = float(config['base']['transaction_cost'])
risk_free_rate = float(config['base']['risk_free_rate'])

total_asset = float(config['allocation']['total_asset'])
risk_factor = float(config['allocation']['risk_factor'])
strategy_risk_factor = float(config['allocation']['strategy_risk_factor'])
stock_risk_factor = float(config['allocation']['stock_risk_factor'])

portfolio_intraday_init_asset = float(config['portfolio_threshold']['portfolio_intraday_init_asset'])

backtesting_start_date = datetime.strptime(config['base']['backtesting_start_date'], '%Y-%m-%d') 

from backtesting.strategy import Strategy
from backtesting.portfolio import Portfolio_intraday, Portfolio_interday, Portfolio
from backtesting.evaluate import get_MDD
from backtesting.evaluate import evaluate_portfolio
from backtesting.Config import Config
from backtesting.portfolio import portfolio_generator


limited_hm_threshold = 0.9
limited_hm_abs_threshold = 5000000

def portfolio_df_limited_hm(portfolio_df, portfolio_trading_record_df, portfolio_cols):
    portfolio_df = portfolio_df.reset_index(drop=True)
    portfolio_trading_record_df = portfolio_trading_record_df.reset_index(drop=True)

    for portfolio_col in portfolio_cols:

        if portfolio_df[f'{portfolio_col}_holding_money'].max() > limited_hm_abs_threshold:

            limited_hm = portfolio_df[f'{portfolio_col}_holding_money'].replace(0, np.nan).dropna().quantile(limited_hm_threshold)
            
            for i in range(len(portfolio_df)):
                hm = portfolio_df.loc[i, f'{portfolio_col}_holding_money'] 
                hm_ratio = limited_hm / hm
                if hm > 0 and hm_ratio < 1:
                    portfolio_df.loc[i, f'{portfolio_col}_profit'] *= hm_ratio
                    portfolio_df.loc[i, f'{portfolio_col}_holding_money'] = limited_hm

                    date = portfolio_df.loc[i, 'Date']
                    in_datetime = datetime(date.year, date.month, date.day, 13, 30) 
                    trading_period = ((portfolio_trading_record_df['inDate'] <= in_datetime) & (portfolio_trading_record_df['outDate'] >= in_datetime)) | pd.isnull(portfolio_trading_record_df['outDate'])
                    portfolio_trading_record_df.loc[trading_period, 'trading_number'] *= hm_ratio
                    portfolio_trading_record_df.loc[trading_period, 'profit'] *= hm_ratio
                    portfolio_trading_record_df.loc[trading_period, 'returns'] *= hm_ratio

    return portfolio_df, portfolio_trading_record_df

def get_EMDD(period, profit_series):
    emdd = 0
    for index in range(period, len(profit_series)):
        emdd = max(emdd, get_MDD(profit_series[index-period:index]))
    return emdd

def get_portfolio_info_df(portfolio_df, portfolio_trading_record_df, portfolio_cols):
    portfolio_info_list = []

    for portfolio_col in portfolio_cols:
        holding_money_series = portfolio_df[f'{portfolio_col}_holding_money'].dropna()
        holding_money_series = holding_money_series[holding_money_series != 0]
        cur_portfolio_trading_record = portfolio_trading_record_df[portfolio_trading_record_df['strategy'] == portfolio_col]
        portfolio_info_list.append([portfolio_col, round(portfolio_df[f'{portfolio_col}_SLR'].iloc[-1], 2),
                                     round(get_MDD(portfolio_df[f'{portfolio_col}_profit']) /10000, 2),
                                   f'{round(holding_money_series.quantile(0.15) / 10000)} ~ {round(holding_money_series.quantile(0.85) / 10000)}',
                                   round(holding_money_series.max() / 10000),
        round((cur_portfolio_trading_record['profit'] * 10000 / ( cur_portfolio_trading_record['inPrice'] * cur_portfolio_trading_record['trading_number'] * 1000 * cur_portfolio_trading_record['holding_days'])).mean(), 2)])
        
    portfolio_info_df = pd.DataFrame(portfolio_info_list, columns=['策略', '槓桿率', '期望最大回撤(萬)', '持倉水位區間(萬)', '最大持倉金額(萬)', '單筆交易期望獲利(每萬/天)'])
    
    return portfolio_info_df

def compute_slr(portfolio_report, portfolio_trading_record, portfolio_cols, total_asset, risk_factor, strategy_risk_factor, risk_period, slr_limit = True):
    portfolio_df = portfolio_report.copy().reset_index(drop=True)
    portfolio_trading_record_df = portfolio_trading_record.copy()

    portfolio_df, portfolio_trading_record_df = portfolio_df_limited_hm(portfolio_df, portfolio_trading_record_df, portfolio_cols)

    start_date = portfolio_df['Date'].iloc[0]

    ori_portfolio_df = portfolio_df[portfolio_df['Date'] >= backtesting_start_date].copy()
    ori_portfolio_trading_record_df = portfolio_trading_record_df[portfolio_trading_record_df['inDate'] >= backtesting_start_date].copy()

    portfolio_compare = []
    portfolio_compare.append(['Original', round(ori_portfolio_df['profit'].sum()/10000, 2), 
    round(get_MDD(ori_portfolio_df['profit'])/10000, 2),
    round(ori_portfolio_df['holding_money'].cummax().iloc[-1]/10000, 2) ])
    
    #ori_portfolio_info_df = get_portfolio_info_df(portfolio_df[portfolio_df['Date'] >= backtesting_start_date],
    # portfolio_trading_record_df[portfolio_trading_record_df['inDate'] >= backtesting_start_date], portfolio_cols)

    for portfolio_col in portfolio_cols:
            
        E_MDD = []
        mdd = 0
        for index in range(risk_period, len(portfolio_df)):
            mdd = max(mdd, get_MDD(portfolio_df[f'{portfolio_col}_profit'][index-risk_period:index]))
            E_MDD.append(mdd) 
            
        portfolio_df[f'{portfolio_col}_EMDD'] = [np.nan] * risk_period + E_MDD
        
    portfolio_df = portfolio_df.iloc[risk_period:].reset_index(drop=True)

    strategy_asset = (risk_factor * total_asset) / len(portfolio_cols)

    for portfolio_col in portfolio_cols:
        portfolio_df[f'{portfolio_col}_SLR'] =  (strategy_risk_factor / portfolio_df[f'{portfolio_col}_EMDD']) * strategy_asset
        if slr_limit:
            portfolio_df.loc[portfolio_df[f'{portfolio_col}_SLR'] > 1, f'{portfolio_col}_SLR'] = 1
        portfolio_df[f'{portfolio_col}_profit'] = portfolio_df[f'{portfolio_col}_profit'] * portfolio_df[f'{portfolio_col}_SLR'].shift(1)
        portfolio_df[f'{portfolio_col}_holding_money'] = portfolio_df[f'{portfolio_col}_holding_money'] * portfolio_df[f'{portfolio_col}_SLR'].shift(1)
        
    portfolio_trading_record_df = portfolio_trading_record_df[portfolio_trading_record_df['inDate'] >= portfolio_df['Date'].iloc[0]]
    portfolio_trading_record_df = portfolio_trading_record_df[portfolio_trading_record_df['strategy'].isin(portfolio_cols)]

    for index in portfolio_trading_record_df.index:
        strategy_name = portfolio_trading_record_df['strategy'].loc[index]
        cur_date = portfolio_trading_record_df['inDate'].loc[index]

        cur_SLR = portfolio_df[portfolio_df['Date'] <= cur_date][f'{strategy_name}_SLR'].iloc[-1]
        portfolio_trading_record_df.loc[index, 'trading_number'] *= cur_SLR
        portfolio_trading_record_df.loc[index, 'returns'] *= cur_SLR
        portfolio_trading_record_df.loc[index, 'profit'] *= cur_SLR

    portfolio_df = portfolio_df[portfolio_df['Date'] >= backtesting_start_date]
    portfolio_trading_record_df = portfolio_trading_record_df[portfolio_trading_record_df['inDate'] >= backtesting_start_date]

    portfolio_slr = Portfolio(name = f'portfolio_SLR_RF{risk_factor}_SRF{strategy_risk_factor}_Period{risk_period}',
                            portfolio_report = portfolio_df, portfolio_trading_record = portfolio_trading_record_df, portfolio_cols = portfolio_cols)
    portfolio_slr.get_plotly_html()

    portfolio_profit_cols = [col + '_profit' for col in portfolio_cols] 
    portfolio_holding_money_cols = [col + '_holding_money' for col in portfolio_cols] 
    portfolio_compare.append(['Adjusted', round(portfolio_df[portfolio_profit_cols].sum(axis=1).sum()/10000, 2),
    round(get_MDD(portfolio_df[portfolio_profit_cols].sum(axis=1))/10000, 2),
    round(portfolio_df[portfolio_holding_money_cols].sum(axis=1).cummax().iloc[-1]/10000, 2) ])
    portfolio_compare_df = pd.DataFrame(portfolio_compare, columns = ['投組狀態', ' 淨利(萬)', '最大回撤金額(萬)', '最大持倉金額(萬)'])
    
    portfolio_info_df = get_portfolio_info_df(portfolio_df, portfolio_trading_record_df, portfolio_cols)

    return portfolio_slr, portfolio_df, portfolio_trading_record_df, portfolio_compare_df, portfolio_info_df
    

