import numpy as np
import pandas as pd
import datetime
import glob
from tqdm import tqdm
import os
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF8')
import warnings
warnings.filterwarnings("ignore")

file_type = '.xlsx'

import configparser

config = configparser.ConfigParser()
config.read('strategy_config.ini')

strategy_path = config['base']['path']
report_path = config['base']['report_path']
strategy_report_paths = glob.glob(strategy_path + '*' + file_type)
intraday_start_day = datetime.datetime.strptime(config['base']['intraday_start_day'], '%Y-%m-%d')
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

def Get_strategy_report(strategy_report_path):
    
    strategy_name = strategy_report_path[len(strategy_path):-len(file_type)]

    stat_df = pd.read_excel(strategy_report_path, sheet_name='整體統計')
    stat_df = stat_df.iloc[:35, :3]
    stat_df.columns = ['statistics', 'value', 'ratio']

    if stat_df['statistics'].iloc[4] != '交易腳本':
        raise ValueError(strategy_name +  ' 非交易腳本 ! ')
        
    trading_df = pd.read_excel(strategy_report_path, sheet_name='交易分析')
    trading_df['進場時間'] = pd.to_datetime(trading_df['進場時間'])
    trading_df['獲利金額'] = trading_df['獲利金額'].replace('[\$,]', '', regex=True).astype(float)
    
    backtesting_trading_days = stat_df[stat_df['statistics'] == '回測K線根數']['value'].values[0]
    backtesting_date = stat_df[stat_df['statistics'] == '回測資料範圍']['value'].values[0]
    #backtesting_date = backtesting_date[:23]
    backtesting_years = round(backtesting_trading_days / year_days, 2)

    net_profit= transform_money_to_float(stat_df[stat_df['statistics'] == '淨利']['value'].values[0])
    mean_holding_days = float(stat_df[stat_df['statistics'] == '全部交易的平均持倉K線根數']['value'].values[0])
    profit_factor = float(stat_df[stat_df['statistics'] == '獲利因子']['value'].values[0])
    total_trading_count = stat_df[stat_df['statistics'] == '總交易次數']['value'].values[0]
    max_position = transform_money_to_float(stat_df[stat_df['statistics'] == '最大投入金額']['value'].values[0])
    total_trading_count_per_year = round(total_trading_count / backtesting_years, 2)
    mean_profit_and_coss_ratio = float(stat_df[stat_df['statistics'] == '平均獲利虧損比']['value'].values[0])
    #win_ratio = float(stat_df[stat_df['statistics'] == '勝率']['value'].values[0][:-1])
    win_loss_series = (trading_df.groupby(['進場時間', '商品名稱'])['獲利金額'].sum() > 0)
    win_ratio = round(win_loss_series.value_counts()[True] / win_loss_series.count() * 100, 2)
    TWRR = float(stat_df[stat_df['statistics'] == '時間加權報酬']['ratio'].values[0][:-1])
    max_holding_irr = float(stat_df[stat_df['statistics'] == '最大投入報酬率']['ratio'].values[0][:-1])
    CAGR = round((pow((float(stat_df[stat_df['statistics'] == '淨利']['ratio'].values[0][:-1]) / 100 + 1), 1 / backtesting_years) - 1)*100, 2)
    MDD = -float(stat_df[stat_df['statistics'] == '最大區間虧損']['ratio'].values[0][:-1])
    risk_reward_ratio_per_year = round(TWRR / backtesting_years / MDD, 2)
    risk_reward_ratio = round((net_profit / max_position) / (MDD/100), 2)

    enough_tradingday_count_per_month = (trading_df.drop_duplicates(subset=['商品名稱', '進場時間']).groupby(trading_df.drop_duplicates(subset=['商品名稱', '進場時間'])['進場時間'].dt.strftime('%Y%m')).count()['進場時間'] > intraday_tradingday_count_per_month_threshold).all()

    max_continue_loss_count, loss_years = eval_daily_report(strategy_report_path)

    strategy_report_list = [strategy_name, backtesting_date, backtesting_years, mean_holding_days, profit_factor, 
                            mean_profit_and_coss_ratio, total_trading_count, total_trading_count_per_year, win_ratio,
                           TWRR, max_holding_irr, CAGR, MDD, risk_reward_ratio_per_year, risk_reward_ratio,
                            max_continue_loss_count, loss_years, enough_tradingday_count_per_month]
    
    return strategy_report_list


def Check_strategy_list(leaderboard_df):
 
    intraday_check_condition = (leaderboard_df['回測年數'] > intraday_backtesting_years_threshold) & \
        (leaderboard_df['風險報酬比'] > intraday_risk_reward_ratio_threshold) & \
        (leaderboard_df['獲利因子'] > intraday_profit_factor_threshold) & (leaderboard_df['最大區間虧損率%'] < intraday_MDD_threshold) & \
        (leaderboard_df['總交易次數(一年平均)'] > intraday_tradingday_count_threshold) & leaderboard_df[f'每月至少{intraday_tradingday_count_per_month_threshold}交易日有訊號']
        
    interday_check_condition =  (leaderboard_df['回測年數'] > interday_backtesting_years_threshold) & \
        (leaderboard_df['風險報酬比'] > interday_risk_reward_ratio_threshold) & \
        (leaderboard_df['獲利因子'] > interday_profit_factor_threshold) & (leaderboard_df['最大區間虧損率%'] < interday_MDD_threshold)

    intraday_df = leaderboard_df[leaderboard_df['平均持倉天數']<=1]
    interday_df = leaderboard_df[leaderboard_df['平均持倉天數']>1]
    
    intraday_df = intraday_df[intraday_check_condition]
    interday_df = interday_df[interday_check_condition]

    return  pd.concat([intraday_df, interday_df])
    

def Compute_corr(leaderboard_names, qualified):
        
    daily_report = pd.read_csv('C:\\Users\\012480\\Documents\\Data\\trading_day_date.txt', parse_dates=['Date'])

    for leaderboard_name in leaderboard_names:
        
        daily_report_df = pd.read_excel(strategy_path + leaderboard_name + file_type, sheet_name='每日報表')
        daily_report_df['Date'] = pd.to_datetime(daily_report_df['日期']) 
        daily_report_df[leaderboard_name] = daily_report_df['獲利'].replace('[\$,]', '', regex=True).astype(float)
        
        daily_report = pd.merge(daily_report, daily_report_df[['Date', leaderboard_name]], on=['Date'], how='outer')
    
    daily_report = daily_report.dropna(subset = daily_report.columns[1:] , how='all')
    #daily_report = daily_report.iloc[:,1:]
    #daily_report = daily_report.loc[~(daily_report==0).all(axis=1)]
    
    daily_report.corr().to_csv(report_path + f'豐神榜{qualified}_相關性.csv', encoding='big5')


if __name__ == '__main__':
    leaderboard_list = []
    for strategy_report_path in tqdm(strategy_report_paths):
        try:
            strategy_report_list = Get_strategy_report(strategy_report_path)
            leaderboard_list.append(strategy_report_list)
        except:
            print(strategy_report_path, ' Fail ! ')

    leaderboard_df = pd.DataFrame(leaderboard_list)
    
    leaderboard_df.columns = ['策略名稱', '回測資料範圍', '回測年數', '平均持倉天數', '獲利因子', '平均獲利虧損比', '總交易次數',
    '總交易次數(一年平均)', '勝率%', '總報酬率%(時間加權)', '最大投入報酬率%', '年化報酬率%', '最大區間虧損率%', '最大區間虧損報酬比(風報比)(單年平均)', '風險報酬比',
    '最大連續虧損月份', '賠錢年份', f'每月至少{intraday_tradingday_count_per_month_threshold}交易日有訊號']
    leaderboard_df.to_excel(report_path + "豐神榜.xlsx", sheet_name='豐神榜績效(XQ)', index = 0)  
    
    qualified_leaderboard_df = Check_strategy_list(leaderboard_df)

    qualified_leaderboard_df.to_excel(report_path + "豐神榜_合格.xlsx", sheet_name='豐神榜績效(XQ)', index = 0)  

    print('Compute correlation ...')
    Compute_corr(qualified_leaderboard_df['策略名稱'], '_合格')
    Compute_corr(leaderboard_df['策略名稱'], '')

    print('Complete ! ')

    os.system("pause")

