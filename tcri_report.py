
import numpy as np
import pandas as pd
import glob
#from tqdm.auto import tqdm
import os
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
#import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
#from dash import Dash, html, dcc
import tkinter
import json
from tqdm import tqdm
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF8')

import warnings
warnings.filterwarnings("ignore")

from backtesting.report import Get_strategy_report, Check_strategy_list, Compute_corr, get_performance
from backtesting.evaluate import get_MDD
import configparser

config = configparser.ConfigParser()
config.read('strategy_config.ini')

strategy_path = config['base']['path']
file_type = config['base']['file_type']
strategy_report_paths = glob.glob(strategy_path + '*' + file_type)
intraday_tradingday_count_per_month_threshold = int(config['intraday_threshold']['tradingday_count_per_month'])
tcri_risk_path = config['base']['tcri_risk_path']
tcri_report_path = config['base']['tcri_report_path']


def get_daily_report_trading_df(strategy_report_path):
    
    daily_report_df = pd.read_excel(strategy_report_path, sheet_name='每日報表')
    daily_report_df['Date'] = pd.to_datetime(daily_report_df['日期']) 
    daily_report_df['獲利'] = daily_report_df['獲利'].replace('[\$,]', '', regex=True).astype(float)
        
    trading_df = pd.read_excel(strategy_report_path, sheet_name='交易分析')
    trading_df['進場時間'] = pd.to_datetime(trading_df['進場時間'])
    trading_df['出場時間'] = pd.to_datetime(trading_df['出場時間'].replace('--', ''))
    trading_df['獲利金額'] = trading_df['獲利金額'].replace('[\$,]', '', regex=True).astype(float)
    trading_df['買進金額'] = trading_df['進場價格'] * trading_df['交易數量'] * 1000
    trading_df = trading_df.sort_values(by='進場時間')
    
    return daily_report_df, trading_df
    

def get_tcri_df(tcri_path):
    
    with open(tcri_path, "r") as f:
        tcri_json = json.load(f)
    
    tcri_list = []
    for date_index in tcri_json.keys():
        tcri_list.append([ datetime.strptime(date_index, "%Y-%m-%d %H:%M:%S"), [stock_id[:4] for stock_id in tcri_json[date_index]]])
        
    tcri_df = pd.DataFrame(tcri_list)
    tcri_df.columns = ['date', 'stocks']

    return tcri_df


def get_tcri_maintain_ratio(strategy_report_paths):

    tcri_df = get_tcri_df(tcri_risk_path)
    tcri_stat_list = []

    for strategy_report_path in strategy_report_paths:
        
        daily_report_df, trading_df = get_daily_report_trading_df(strategy_report_path)
        
        ori_trading_num = len(trading_df)
        tcri_trading_num = 0

        for index in range(len(trading_df)):
            cur_stock_id = trading_df['商品名稱'].iloc[index][-8:-4]
            cur_date = trading_df['進場時間'].iloc[index]
            
            tcri_stocks = tcri_df[tcri_df['date'] <= cur_date].iloc[0]['stocks']

            if cur_stock_id not in tcri_stocks:
                tcri_trading_num += 1
            
        tcri_stat_list.append([strategy_report_path.split('\\')[1][:-5], ori_trading_num, tcri_trading_num, tcri_trading_num/ ori_trading_num ])
        
    tcri_stat_df = pd.DataFrame(tcri_stat_list)
    tcri_stat_df.columns = ['策略', '交易次數', '交易次數(剔除TCRI)', '保留比率']
    
    return tcri_stat_df


def get_tcri_trading_df(trading_df):

    tcri_df = get_tcri_df(tcri_risk_path)
    tcri_index = []

    for index in range(len(trading_df)):
        cur_stock_id = trading_df['商品名稱'].iloc[index][-8:-4]
        cur_date = trading_df['進場時間'].iloc[index]
        
        tcri_stocks = tcri_df[tcri_df['date'] <= cur_date].iloc[0]['stocks']

        if cur_stock_id not in tcri_stocks:
            tcri_index.append(index)
            
    return trading_df[trading_df.index.isin(tcri_index)]


def get_tcri_compare():

    get_tcri_maintain_ratio(strategy_report_paths).to_excel(tcri_report_path + "保留比率.xlsx", sheet_name='保留比率', index = 0)  

    leaderboard_list = []

    for strategy_report_path in tqdm(strategy_report_paths):
        
        #try:
            
            daily_report_df, trading_df = get_daily_report_trading_df(strategy_report_path)

            strategy_report_list = get_performance(strategy_report_path, daily_report_df, trading_df)
            strategy_tcri_report_list = get_performance(strategy_report_path, daily_report_df, get_tcri_trading_df(trading_df))
            
            leaderboard_list.append(strategy_report_list)
            leaderboard_list.append(strategy_tcri_report_list)
            leaderboard_list.append([np.nan]*len(strategy_report_list))

        #except:
        #    print(strategy_report_path, ' Fail ! ')

    leaderboard_df = pd.DataFrame(leaderboard_list)

    leaderboard_df.columns = ['策略名稱', '回測資料範圍', '回測年數', '平均持倉天數', '獲利因子', '總交易次數',
    '總交易次數(一年平均)', '勝率%', '最大投入報酬率%', '最大持倉金額(萬)', '最大區間虧損率%', '風險報酬比',
    '最大連續虧損月份', '賠錢年份', f'每月至少{intraday_tradingday_count_per_month_threshold}交易日有訊號', '平均每月交易日', 'A\n(X=總收/總損)\nA=(X-1)/(X+1)', 'B\n(Y=均正報酬/均負報酬)\nB=(Y-1)/(Y+1)*勝率', 'C\n(Z=平均單筆賺賠)\nC=Z/MDD', 'A+B+C']

    leaderboard_df.to_excel(tcri_report_path + "豐神榜.xlsx", sheet_name='豐神榜績效(XQ)', index = 0)  
