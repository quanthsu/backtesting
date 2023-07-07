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

from . import analytics
from . import plot
from backtesting.tcri_report import get_tcri_trading_df


class Strategy(analytics.Analytics, plot.Plot):
    
    def __init__(self, strategy_report_path = None, daily_report_df = None, trading_record_df = None, tcri_limit = False):
        
        self.tcri_limit = tcri_limit

        config = configparser.ConfigParser()
        config.read('strategy_config.ini')

        strategy_path = config['base']['path']
        report_path = config['base']['report_path']
        self.report_plot_path = config['base']['report_plot_path']
        file_type = config['base']['file_type']
        year_days = int(config['base']['year_days'])
        transaction_cost = float(config['base']['transaction_cost'])
        strategy_init_asset = int(config['base']['strategy_init_asset'])

        if strategy_report_path is not None:
            #self.name = strategy_report_path[len(strategy_path):-len(file_type)]
            self.name = strategy_report_path.split('\\')[-1][:-len(file_type)]

        if trading_record_df is None:
            trading_record_df = pd.read_excel(strategy_report_path, sheet_name='交易分析')
            if self.tcri_limit:
                trading_record_df = get_tcri_trading_df(trading_record_df)
            trading_record_df = trading_record_df[['商品名稱', '進場時間', '進場方向', '進場價格', '出場時間', '出場價格', '持有區間', '交易數量', '獲利金額', '報酬率']]
            trading_record_df.columns = ['ticker', 'inDate', 'direction', 'inPrice', 'outDate', 'outPrice', 'holding_days', 'trading_number', 'profit', 'returns']
            trading_record_df['profit'] = trading_record_df['profit'].replace('[\$,]', '', regex=True).astype(float)
            trading_record_df['returns'] = trading_record_df['profit']
            trading_record_df['inDate'] = pd.to_datetime(trading_record_df['inDate'])
            trading_record_df['outDate'] = pd.to_datetime(trading_record_df['outDate'].replace('--', ''))

        if daily_report_df is None:
            daily_report_df = pd.read_excel(strategy_report_path, sheet_name='每日報表')
            daily_report_df = daily_report_df[daily_report_df['最大投入金額'] != '0'].reset_index(drop=True)
            daily_report_df['Date'] = pd.to_datetime(daily_report_df['日期']) 
            daily_report_df['profit'] = daily_report_df['獲利'].replace('[\$,]', '', regex=True).astype(float)
            if self.tcri_limit:  
                daily_report_df['profit'] = 0
                for i in range(len(trading_record_df)):
                    out_date = trading_record_df['outDate'].iloc[i].date()
                    daily_report_df.loc[daily_report_df['Date'].dt.date == out_date, 'profit'] += trading_record_df.loc[i, 'profit']

            #daily_report_df['max_holding_money'] = daily_report_df['最大投入金額'].replace('[\$,]', '', regex=True).astype(float)

        #self.max_holding_money = daily_report_df['max_holding_money'].iloc[-1]
        datetime_list = [daily_report_df['Date'].iloc[0] - timedelta(days=1)] + list(daily_report_df['Date'])
        nav_list = [1] + list((daily_report_df['profit'].cumsum() + strategy_init_asset) / strategy_init_asset)
        nav_df = pd.DataFrame()
        nav_df['nav'] = nav_list
        nav_df.index = datetime_list

        self.daily_report = daily_report_df
        self.navRecords = nav_df
        self.tradingRecords = trading_record_df
        self.costs = transaction_cost / 100
        self.date_list = nav_df.index



