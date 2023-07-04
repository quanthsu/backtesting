
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
merge_strategy_path = config['base']['merge_path']
report_path = config['base']['report_path']
report_plot_path = config['base']['report_plot_path']
file_type = config['base']['file_type']
strategy_report_paths = glob.glob(strategy_path + '*' + file_type)
intraday_start_day = datetime.strptime(config['base']['intraday_start_day'], '%Y-%m-%d')
year_days = int(config['base']['year_days'])
transaction_cost = float(config['base']['transaction_cost'])

merge_folders = glob.glob(f'./{merge_strategy_path}/*')

for merge_folder in merge_folders:
    merge_files = glob.glob(merge_folder + '/*')
    name = merge_folder.split('\\')[-1]
    daily_report_df = pd.DataFrame()
    trading_record_df = pd.DataFrame()
    
    for merge_file in merge_files:
        
        daily_report =  pd.read_excel(merge_file, sheet_name='每日報表')
        daily_report = daily_report[daily_report['最大投入金額'] != '0']
        trading_record = pd.read_excel(merge_file, sheet_name='交易分析')

        daily_report_df = pd.concat([daily_report_df, daily_report])
        trading_record_df = pd.concat([trading_record_df, trading_record])

    daily_report_df['日期'] = pd.to_datetime(daily_report_df['日期']) 
    daily_report_df = daily_report_df.sort_values(by=['日期'])
    daily_report_df['最大投入金額'] = daily_report_df['最大投入金額'].replace('[\$,]', '', regex=True).astype(float)
    daily_report_df = daily_report_df.drop_duplicates(subset=['日期', '獲利'])
    daily_report_df = daily_report_df[['日期', '獲利', '最大投入金額', '商品檔數']]
    daily_report_df = daily_report_df.reset_index(drop=True)
    daily_report_df['最大投入金額'] = daily_report_df['最大投入金額'].cummax()
    daily_report_df = daily_report_df.drop_duplicates(subset=['日期'])

    trading_record_df['進場時間'] = pd.to_datetime(trading_record_df['進場時間']) 
    trading_record_df = trading_record_df.sort_values(by=['進場時間'])
    trading_record_df = trading_record_df.reset_index(drop=True)

    with pd.ExcelWriter(strategy_path + f'{name}.xlsx') as writer:  
        daily_report_df.to_excel(writer, sheet_name='每日報表', index=None)
        trading_record_df.to_excel(writer, sheet_name='交易分析', index=None)
        