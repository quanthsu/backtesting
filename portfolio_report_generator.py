
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
from backtesting.portfolio import portfolio_generator

if __name__ == '__main__':

    portfolio, portfolio_df, portfolio_trading_record_df, portfolio_cols = portfolio_generator('portfolio', 'portfolio_interday', 'portfolio_intraday', tcri_limit=False)
    portfolio_tcri, portfolio_df_tcri, portfolio_trading_record_df_tcri, portfolio_cols_tcri = portfolio_generator('portfolio_TCRI', 'portfolio_interday_TCRI', 'portfolio_intraday_TCRI', tcri_limit=True)

    #print('\r Generating intraday report ...', end = '', flush=True)
    #print('\r Generating interday report ...', end = '', flush=True)
    
    print('\r Generating portfolio report ...')
    portfolio.get_plotly_html()
    portfolio_tcri.get_plotly_html()

    print('Complete ! ')

    os.system("pause")