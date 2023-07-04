
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
strategy_report_paths = glob.glob(strategy_path + '*' + file_type)
intraday_start_day = datetime.strptime(config['base']['intraday_start_day'], '%Y-%m-%d')
year_days = int(config['base']['year_days'])
transaction_cost = float(config['base']['transaction_cost'])

portfolio_intraday_init_asset = float(config['portfolio_threshold']['portfolio_intraday_init_asset'])

from backtesting.strategy import Strategy
from backtesting.portfolio import Portfolio_intraday, Portfolio_interday, Portfolio

portfolio_interday_paths = glob.glob('./portfolio_interday/' + '*' + file_type)
portfolio_intraday_paths = glob.glob('./portfolio_intraday/' + '*' + file_type)

if __name__ == '__main__':

    portfolio_cols = []
    if portfolio_intraday_paths:
        print('\r Preparing intraday ...', end = '', flush=True)
        portfolio_intraday = Portfolio_intraday(name = 'portfolio_intraday', portfolio_intraday_init_asset = portfolio_intraday_init_asset, portfolio_path = portfolio_intraday_paths)
        portfolio_intraday.get_plotly_html()
        portfolio_cols += portfolio_intraday.portfolio_cols
        portfolio_df = portfolio_intraday.daily_report
        portfolio_trading_record_df = portfolio_intraday.tradingRecords

    if portfolio_interday_paths:
        print('\r Preparing interday ...')
        portfolio_interday = Portfolio_interday(name = 'portfolio_interday', portfolio_path = portfolio_interday_paths)
        portfolio_interday.get_plotly_html()
        portfolio_cols += portfolio_interday.portfolio_cols
        portfolio_df = portfolio_interday.daily_report
        portfolio_trading_record_df = portfolio_interday.tradingRecords

    if portfolio_intraday_paths and portfolio_interday_paths:
        portfolio_df = pd.merge(portfolio_intraday.daily_report, portfolio_interday.daily_report, on=['Date'], how='outer')
        portfolio_df = portfolio_df.sort_values(by = ['Date']).reset_index(drop=True)
        portfolio_trading_record_df = pd.concat([portfolio_intraday.tradingRecords, portfolio_interday.tradingRecords]).sort_values(by = ['inDate']).reset_index(drop=True)
    
    portfolio = Portfolio(portfolio_report = portfolio_df, portfolio_trading_record = portfolio_trading_record_df, portfolio_cols = portfolio_cols)

    #print('\r Generating intraday report ...', end = '', flush=True)
    #print('\r Generating interday report ...', end = '', flush=True)
    
    print('\r Generating portfolio report ...')
    portfolio.get_plotly_html()
    
    print('Complete ! ')

    os.system("pause")