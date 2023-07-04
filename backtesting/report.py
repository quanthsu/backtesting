import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import glob
from tqdm import tqdm
import os
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF8')
from backtesting.evaluate import get_MDD
#import warnings
#warnings.filterwarnings("ignore")

import configparser

config = configparser.ConfigParser()
config.read('strategy_config.ini')

strategy_path = config['base']['path']
report_path = config['base']['report_path']
trading_day_date_path = config['base']['trading_day_date_path']
file_type = config['base']['file_type']
strategy_report_paths = glob.glob(strategy_path + '*' + file_type)
intraday_start_day = datetime.strptime(config['base']['intraday_start_day'], '%Y-%m-%d')
year_days = int(config['base']['year_days'])

continue_loss_month_threshold = int(config['portfolio_threshold']['continue_loss_month'])

intraday_backtesting_years_threshold = float(config['intraday_threshold']['backtesting_years'])
intraday_risk_reward_ratio_threshold = float(config['intraday_threshold']['risk_reward_ratio'])
intraday_profit_factor_threshold = float(config['intraday_threshold']['profit_factor'])
intraday_MDD_threshold = float(config['intraday_threshold']['MDD'])
intraday_tradingday_count_threshold = int(config['intraday_threshold']['tradingday_count'])
intraday_tradingday_count_per_month_threshold = int(config['intraday_threshold']['tradingday_count_per_month'])

interday_backtesting_years_threshold = float(config['interday_threshold']['backtesting_years'])
interday_risk_reward_ratio_threshold = float(config['interday_threshold']['risk_reward_ratio'])
interday_profit_factor_threshold = float(config['interday_threshold']['profit_factor'])
interday_MDD_threshold = float(config['interday_threshold']['MDD'])


def eval_daily_report(strategy_report_path):
    daily_report_df = pd.read_excel(strategy_report_path, sheet_name='每日報表')
    daily_report_df['Date'] = pd.to_datetime(daily_report_df['日期']) 
    daily_report_df['profit'] = daily_report_df['獲利'].replace('[\$,]', '', regex=True).astype(float)

    continue_loss_count = 0
    max_continue_loss_count = 0

    for profit in daily_report_df.groupby(daily_report_df['Date'].dt.strftime('%Y%m'))['profit'].sum():
        if profit < 0:
            continue_loss_count += 1
            max_continue_loss_count = max(max_continue_loss_count, continue_loss_count)
        else:
            continue_loss_count = 0

    daily_report_year_profit_df = daily_report_df.groupby(daily_report_df['Date'].dt.strftime('%Y'))['profit'].sum()
    loss_years = daily_report_year_profit_df[daily_report_year_profit_df<0].index

    #over_max_continue_loss_count = max_continue_loss_count <= continue_loss_month_threshold
    #loss_years_or_not = True if len(loss_years) == 0 else False

    return max_continue_loss_count, loss_years.values


def transform_money_to_float(money):
    return locale.atof(money.strip("$"))


def get_daily_df(trading_df):

    start_date = trading_df['進場時間'].min() 
    end_date = trading_df['進場時間'].max() 
    start_date = date(start_date.year, start_date.month, start_date.day)
    end_date = date(end_date.year, end_date.month, end_date.day)

    cur_date = start_date 
    money_list = []

    while cur_date <= end_date:
        in_datetime = datetime(cur_date.year, cur_date.month, cur_date.day, 13, 30)
        #cur_trading_df = trading_df[(trading_df['進場時間'].dt.date <= cur_date) & (trading_df['出場時間'].dt.date >= cur_date)]
        cur_trading_df = trading_df[(trading_df['進場時間'] <= in_datetime) & (trading_df['出場時間'] >= in_datetime)]
        
        money_list.append([cur_date, trading_df[trading_df['出場時間'].dt.date == cur_date]['獲利金額'].sum(), cur_trading_df['買進金額'].sum()])
        cur_date += timedelta(days=1)

    daily_money_df = pd.DataFrame(money_list)
    daily_money_df.columns = ['date', 'profit', 'holding_money']

    return daily_money_df
    
def get_performance(strategy_report_path, daily_report_df, trading_df):
    trading_df = trading_df.copy()
    daily_money_df = get_daily_df(trading_df)
    
    #backtesting_trading_days = stat_df[stat_df['statistics'] == '回測K線根數']['value'].values[0]
    #backtesting_date = stat_df[stat_df['statistics'] == '回測資料範圍']['value'].values[0]
    backtesting_date = f"{daily_report_df['Date'].min().date()}~{daily_report_df['Date'].max().date()}"
    #backtesting_date = backtesting_date[:23]
    #backtesting_years = round((datetime.strptime(backtesting_date[23-10:23], "%Y/%m/%d") - datetime.strptime(backtesting_date[:10], "%Y/%m/%d")) / timedelta(days=365), 2)
    backtesting_years = round((datetime.strptime(backtesting_date.split('~')[1], "%Y-%m-%d") - datetime.strptime(backtesting_date.split('~')[0], "%Y-%m-%d")) / timedelta(days=365), 2)

    net_profit = trading_df['獲利金額'].sum()
    mean_holding_days = round((trading_df['出場時間'] - trading_df['進場時間']).mean() / timedelta(days=1), 2)
    profit_factor = -np.sum(trading_df[trading_df['獲利金額'] > 0]['獲利金額']) / np.sum(trading_df[trading_df['獲利金額'] < 0]['獲利金額'])
    total_trading_count = len(trading_df.groupby(['進場時間', '商品名稱']))
    #max_position = daily_money_df['holding_money'].cummax()[len(daily_money_df)-1]
    max_position = daily_money_df['holding_money'].max()
    total_trading_count_per_year = round(total_trading_count / backtesting_years, 2)
    #mean_profit_and_coss_ratio = float(stat_df[stat_df['statistics'] == '平均獲利虧損比']['value'].values[0])
    win_loss_series = (trading_df.groupby(['進場時間', '商品名稱'])['獲利金額'].sum() > 0)
    win_ratio = round(win_loss_series.value_counts()[True] / win_loss_series.count() * 100, 2)
    #TWRR = float(stat_df[stat_df['statistics'] == '時間加權報酬']['ratio'].values[0][:-1])
    max_holding_irr = round(net_profit / max_position * 100, 2) 
    #CAGR = round((pow((net_profit / max_position + 1), 1 / backtesting_years) - 1)*100, 2)
    #MDD = -transform_money_to_float(stat_df[stat_df['statistics'] == '最大區間虧損']['value'].values[0])
    MDD = get_MDD(daily_report_df['獲利'])
    MDD_ratio = round(MDD / max_position * 100, 2)
    #risk_reward_ratio_per_year = round(TWRR / backtesting_years / (MDD/max_position*100), 2)
    risk_reward_ratio = round(net_profit / MDD, 2)

    A_ratio = (profit_factor-1)/(profit_factor+1)
    mean_profit_loss_ratio = -(np.mean(trading_df[trading_df['報酬率'] > 0]['報酬率'] )) / (np.mean(trading_df[trading_df['報酬率'] < 0]['報酬率'] ))
    B_ratio = (mean_profit_loss_ratio-1)/(mean_profit_loss_ratio+1) * win_loss_series.value_counts()[True] / win_loss_series.count()
    mean_profit = net_profit / total_trading_count
    C_ratio = mean_profit / MDD

    ABC_ratio = A_ratio + B_ratio + C_ratio

    enough_tradingday_count_per_month = (trading_df.drop_duplicates(subset=['商品名稱', '進場時間']).groupby(trading_df.drop_duplicates(subset=['商品名稱', '進場時間'])['進場時間'].dt.strftime('%Y%m')).count()['進場時間'] > intraday_tradingday_count_per_month_threshold).all()
    
    trading_df['進場時間'] = trading_df['進場時間'].dt.date
    trading_df_drop_duplicates = (trading_df.drop_duplicates(subset=['進場時間']))
    mean_tradingday_count_per_month = round(len(trading_df_drop_duplicates) / (backtesting_years*12), 2)

    max_continue_loss_count, loss_years = eval_daily_report(strategy_report_path)

    strategy_name = strategy_report_path[len(strategy_path):-len(file_type)]
    strategy_report_list = [strategy_name, backtesting_date, backtesting_years, mean_holding_days, round(profit_factor, 2), 
                            total_trading_count, total_trading_count_per_year, win_ratio,
                            max_holding_irr, round(max_position / 10000, 2), MDD_ratio, risk_reward_ratio,
                            max_continue_loss_count, loss_years, enough_tradingday_count_per_month, mean_tradingday_count_per_month,
                            round(A_ratio, 2), round(B_ratio, 2), round(C_ratio, 2), round(ABC_ratio, 2)]
    
    return strategy_report_list


def Get_strategy_report(strategy_report_path):

    #stat_df = pd.read_excel(strategy_report_path, sheet_name='整體統計')
    #stat_df = stat_df.iloc[:35, :3]
    #stat_df.columns = ['statistics', 'value', 'ratio']

    #if stat_df['statistics'].iloc[4] != '交易腳本':
    #    raise ValueError(strategy_name +  ' 非交易腳本 ! ')
        
    daily_report_df = pd.read_excel(strategy_report_path, sheet_name='每日報表')
    daily_report_df['Date'] = pd.to_datetime(daily_report_df['日期']) 
    daily_report_df['獲利'] = daily_report_df['獲利'].replace('[\$,]', '', regex=True).astype(float)

    trading_df = pd.read_excel(strategy_report_path, sheet_name='交易分析')
    trading_df['進場時間'] = pd.to_datetime(trading_df['進場時間'])
    trading_df['出場時間'] = pd.to_datetime(trading_df['出場時間'].replace('--', ''))
    trading_df['獲利金額'] = trading_df['獲利金額'].replace('[\$,]', '', regex=True).astype(float)
    trading_df['買進金額'] = trading_df['進場價格'] * trading_df['交易數量'] * 1000
    trading_df = trading_df.sort_values(by='進場時間')
    
    return get_performance(strategy_report_path, daily_report_df, trading_df)


def Check_strategy_list(leaderboard_df):
 
    intraday_check_condition = (leaderboard_df['回測年數'] >= intraday_backtesting_years_threshold) \
        & (leaderboard_df['風險報酬比'] >= intraday_risk_reward_ratio_threshold) \
        & (leaderboard_df['獲利因子'] >= intraday_profit_factor_threshold) \
        & (leaderboard_df['最大區間虧損率%'] < intraday_MDD_threshold)  \
        & (leaderboard_df['A+B+C'] >= 0.3) \
        #& (leaderboard_df['總交易次數(一年平均)'] >= intraday_tradingday_count_threshold) \
        #& leaderboard_df[f'每月至少{intraday_tradingday_count_per_month_threshold}交易日有訊號'
        
    interday_check_condition =  (leaderboard_df['回測年數'] >= interday_backtesting_years_threshold) & \
        (leaderboard_df['風險報酬比'] >= interday_risk_reward_ratio_threshold) & \
        (leaderboard_df['獲利因子'] >= interday_profit_factor_threshold) & (leaderboard_df['最大區間虧損率%'] < interday_MDD_threshold)

    intraday_df = leaderboard_df[leaderboard_df['平均持倉天數']<=1]
    interday_df = leaderboard_df[leaderboard_df['平均持倉天數']>1]
    
    intraday_df = intraday_df[intraday_check_condition]
    interday_df = interday_df[interday_check_condition]

    return  pd.concat([intraday_df, interday_df])
    

def Compute_corr(leaderboard_names, qualified):
        
    daily_report = pd.read_csv(trading_day_date_path, parse_dates=['Date'])

    for leaderboard_name in leaderboard_names:
        
        daily_report_df = pd.read_excel(strategy_path + leaderboard_name + file_type, sheet_name='每日報表')
        daily_report_df['Date'] = pd.to_datetime(daily_report_df['日期']) 
        daily_report_df[leaderboard_name] = daily_report_df['獲利'].replace('[\$,]', '', regex=True).astype(float)
        
        daily_report = pd.merge(daily_report, daily_report_df[['Date', leaderboard_name]], on=['Date'], how='outer')
    
    daily_report = daily_report.dropna(subset = daily_report.columns[1:] , how='all')
    #daily_report = daily_report.iloc[:,1:]
    #daily_report = daily_report.loc[~(daily_report==0).all(axis=1)]
    
    daily_report.corr().to_csv(report_path + f'豐神榜{qualified}_相關性.csv', encoding='big5')


