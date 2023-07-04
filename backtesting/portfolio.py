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
#from dash import Dash, html, dcc
from backtesting.strategy import Strategy

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
report_plot_portfolio_path = config['base']['report_plot_portfolio_path']
trading_day_date_path = config['base']['trading_day_date_path']
file_type = config['base']['file_type']
year_days = int(config['base']['year_days'])
transaction_cost = float(config['base']['transaction_cost'])

from . import analytics
from . import plot

class Portfolio_interday(analytics.Analytics, plot.Plot_portfolio):
    
    def __init__(self, name = 'portfolio_interday', portfolio_path = None, daily_report_df = None, trading_record_df = None):
        self.name = name
        self.report_plot_path = report_plot_portfolio_path

        portfolio_df = pd.read_csv(trading_day_date_path, parse_dates=['Date'])
        portfolio_trading_record_df = pd.DataFrame()

        profit_col = []
        holding_money_col = []
        all_col = []
        self.portfolio_cols = []

        for strategy_report_path in portfolio_path:
            
            #try:
                strategy = Strategy(strategy_report_path)
                
                daily_report_df = strategy.daily_report
                holding_money_df = strategy._get_daily_holding_money()
                holding_money_df['Date'] = holding_money_df.index
                strategy_df = pd.merge(daily_report_df[['Date', 'profit']], holding_money_df[['Date', 'HoldingMoney']], on=['Date'])
                strategy_trading_df = strategy.tradingRecords
                strategy_trading_df['strategy'] = strategy.name
                self.portfolio_cols.append(strategy.name)
                
                portfolio_df = pd.merge(portfolio_df, strategy_df, on=['Date'], how='outer')
                portfolio_trading_record_df = pd.concat([portfolio_trading_record_df, strategy_trading_df])

                profit_col += [strategy.name + '_profit']
                holding_money_col += [strategy.name + '_holding_money']
                all_col += [strategy.name + '_profit', strategy.name + '_holding_money']
                
            #except:
            #    continue
                
        
        portfolio_trading_record_df = portfolio_trading_record_df.sort_values(by = ['inDate']).reset_index(drop=True)

        portfolio_df.columns = ['Date'] + all_col
        portfolio_df = portfolio_df.dropna(subset = portfolio_df.columns[1:] , how='all')
        portfolio_df['profit'] = portfolio_df[profit_col].sum(axis=1)
        portfolio_df['holding_money'] = portfolio_df[holding_money_col].sum(axis=1) 
        portfolio_df['max_holding_money'] = portfolio_df['holding_money'].cummax()

        self.max_holding_money = portfolio_df['max_holding_money'].iloc[-1]
        datetime_list = [portfolio_df['Date'].iloc[0] - timedelta(days=1)] + list(portfolio_df['Date'])
        nav_list = [1] + list((portfolio_df['profit'].cumsum() + self.max_holding_money) / self.max_holding_money)
        nav_df = pd.DataFrame()
        nav_df['nav'] = nav_list
        nav_df.index = datetime_list

        self.daily_report = portfolio_df
        self.navRecords = nav_df
        self.tradingRecords = portfolio_trading_record_df
        self.costs = transaction_cost / 100
        self.date_list = nav_df.index


class Portfolio_intraday(analytics.Analytics, plot.Plot_portfolio):
    
    def __init__(self, name = 'portfolio_intraday', portfolio_path = None, portfolio_intraday_init_asset = 1000000, daily_report_df = None, trading_record_df = None):
        
        self.name = name
        self.report_plot_path = report_plot_portfolio_path

        def intraday_profit_renew(strategy_df, init_asset):
            for i in range(len(strategy_df)):
                holding_money_ratio = init_asset / strategy_df.loc[i, 'HoldingMoney'] 
                if holding_money_ratio < 1:
                    strategy_df.loc[i, 'profit'] = strategy_df.loc[i, 'profit'] * holding_money_ratio
                    strategy_df.loc[i, 'HoldingMoney'] = init_asset

        portfolio_df = pd.read_csv(trading_day_date_path, parse_dates=['Date'])
        portfolio_trading_record_df = pd.DataFrame()

        profit_col = []
        holding_money_col = []
        all_col = []
        self.portfolio_cols = []

        for strategy_report_path in portfolio_path:
            
            try:
                strategy = Strategy(strategy_report_path)
                
                daily_report_df = strategy.daily_report
                holding_money_df = strategy._get_daily_holding_money()
                holding_money_df['Date'] = holding_money_df.index
                strategy_df = pd.merge(daily_report_df[['Date', 'profit']], holding_money_df[['Date', 'HoldingMoney']], on=['Date'])
                strategy_trading_df = strategy.tradingRecords
                strategy_trading_df['strategy'] = strategy.name
                self.portfolio_cols.append(strategy.name)

                intraday_profit_renew(strategy_df, portfolio_intraday_init_asset)
                portfolio_df = pd.merge(portfolio_df, strategy_df, on=['Date'], how='outer')
                portfolio_trading_record_df = pd.concat([portfolio_trading_record_df, strategy_trading_df])

                profit_col += [strategy.name + '_profit']
                holding_money_col += [strategy.name + '_holding_money']
                all_col += [strategy.name + '_profit', strategy.name + '_holding_money']
                
            except:
                continue

        portfolio_trading_record_df = portfolio_trading_record_df.sort_values(by = ['inDate']).reset_index(drop=True)

        portfolio_df.columns = ['Date'] + all_col
        portfolio_df = portfolio_df.dropna(subset = portfolio_df.columns[1:], how='all')
        portfolio_df['profit'] = portfolio_df[profit_col].sum(axis=1)
        portfolio_df['holding_money'] = portfolio_df[holding_money_col].sum(axis=1)
        portfolio_df['max_holding_money'] = portfolio_df['holding_money'].cummax()

        self.max_holding_money = portfolio_df['max_holding_money'].iloc[-1]
        datetime_list = [portfolio_df['Date'].iloc[0] - timedelta(days=1)] + list(portfolio_df['Date'])
        nav_list = [1] + list((portfolio_df['profit'].cumsum() + self.max_holding_money) / self.max_holding_money)
        nav_df = pd.DataFrame()
        nav_df['nav'] = nav_list
        nav_df.index = datetime_list

        self.daily_report = portfolio_df
        self.navRecords = nav_df
        self.tradingRecords = portfolio_trading_record_df
        self.costs = transaction_cost / 100
        self.date_list = nav_df.index
                

class Portfolio(analytics.Analytics, plot.Plot_portfolio):
    
    def __init__(self, name = 'portfolio', portfolio_report = None, portfolio_trading_record = None, portfolio_cols = None):
        
        self.name = name
        self.report_plot_path = report_plot_portfolio_path
        self.portfolio_cols = portfolio_cols

        portfolio_df = portfolio_report
        if not self.portfolio_cols:
            self.portfolio_cols = list(portfolio_trading_record['strategy'].unique())

        portfolio_profit_cols = [col + '_profit' for col in self.portfolio_cols] 
        portfolio_holding_money_cols = [col + '_holding_money' for col in self.portfolio_cols] 
        portfolio_df['profit'] = portfolio_df[portfolio_profit_cols].sum(axis=1)
        portfolio_df['holding_money'] = portfolio_df[portfolio_holding_money_cols].sum(axis=1)
        portfolio_df['max_holding_money'] = portfolio_df['holding_money'].cummax()

        self.max_holding_money = portfolio_df['max_holding_money'].iloc[-1]
        datetime_list = [portfolio_df['Date'].iloc[0] - timedelta(days=1)] + list(portfolio_df['Date'])
        nav_list = [1] + list((portfolio_df['profit'].cumsum() + self.max_holding_money) / self.max_holding_money)
        nav_df = pd.DataFrame()
        nav_df['nav'] = nav_list
        nav_df.index = datetime_list

        self.daily_report = portfolio_df
        self.navRecords = nav_df
        self.tradingRecords = portfolio_trading_record[portfolio_trading_record['strategy'].isin(self.portfolio_cols)].reset_index(drop=True)
        self.costs = transaction_cost / 100
        self.date_list = nav_df.index
