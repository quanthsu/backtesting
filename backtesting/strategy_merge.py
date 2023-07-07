
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


def get_occupy_ratio(hm):
    hm_line = hm.replace(0, np.nan).dropna().quantile(0.7)
    hm_adj = (hm_line - hm)
    hm_adj = hm_adj[hm_adj > 0]

    return  1 - hm_adj.sum() / (hm_line * len(hm_adj))

def merge_condition_hm_quantile(portfolio_df, pre_st_que, cur_st_que):
                
    #print(pre_st_que, cur_st_que)
    
    pre_profit, pre_hm = get_st_profit_hm(portfolio_df, pre_st_que)
    cur_profit, cur_hm = get_st_profit_hm(portfolio_df, cur_st_que)
    combined_profit, combined_hm = get_st_profit_hm(portfolio_df, pre_st_que + cur_st_que)
    
    pre_MDD, cur_MDD, combined_MDD = get_MDD(pre_profit), get_MDD(cur_profit), get_MDD(combined_profit)
    pre_MDD_ratio, cur_MDD_ratio, combined_MDD_ratio = pre_MDD / pre_hm.quantile(0.9), cur_MDD / cur_hm.quantile(0.9), combined_MDD / combined_hm.quantile(0.9)

    pre_occupy_ratio = get_occupy_ratio(pre_hm)
    cur_occupy_ratio = get_occupy_ratio(cur_hm)
    combined_occupy_ratio = get_occupy_ratio(combined_hm)
    
    #print(pre_occupy_ratio, cur_occupy_ratio, combined_occupy_ratio)

    return combined_occupy_ratio > max(pre_occupy_ratio, cur_occupy_ratio) + 0.05 and combined_MDD < pre_MDD + cur_MDD and \
         (combined_MDD_ratio < pre_MDD_ratio or combined_MDD_ratio < cur_MDD_ratio)

def judge_hm_usage(portfolio_df, pre_st_que, cur_st_que):
    hm_df = pd.DataFrame()
    hm_df['date'] = portfolio_df['Date']

    pre_hm_list = [st + '_holding_money' for st in pre_st_que]
    cur_hm_list = [st + '_holding_money' for st in cur_st_que]

    hm_df['pre_hm'] = portfolio_df[pre_hm_list].sum(axis=1)
    hm_df['cur_hm'] = portfolio_df[cur_hm_list].sum(axis=1)

        
    hm_usage_count = len(hm_df[(hm_df['pre_hm'] != 0) & (hm_df['cur_hm'] != 0)])

    hm_usage_ratio = hm_usage_count / len(hm_df)

    return hm_usage_ratio < 0.01

def merge_condition1(portfolio_df, pre_st_que, cur_st_que):

    max_holding_money_threshold = 1.1
    hm_ratio = 0.5
    mdd_threshold = 1.1
    #print(pre_st_que, cur_st_que)
    pre_np, pre_mdd, pre_max_hm, pre_mean_hm, pre_occupy_ratio = get_st_info(portfolio_df, pre_st_que)
    cur_np, cur_mdd, cur_max_hm, cur_mean_hm, cur_occupy_ratio = get_st_info(portfolio_df, cur_st_que)
    
    combined_np, combined_mdd, combined_max_hm, combined_mean_hm, combined_occupy_ratio = get_st_info(portfolio_df, pre_st_que + cur_st_que)
            
    return pre_max_hm / cur_max_hm > hm_ratio and pre_max_hm / cur_max_hm < 1 + hm_ratio \
    and combined_max_hm < max(pre_max_hm, cur_max_hm) * max_holding_money_threshold \
        and combined_mdd < max(pre_mdd, cur_mdd) * mdd_threshold

def merge_condition_synthesized(portfolio_df, pre_st_que, cur_st_que):

    pre_np, pre_sharpe_ratio, pre_sortino_ratio, pre_mdd_money, pre_mdd_ratio, pre_max_hm, pre_occupy_ratio, pre_vol, pre_beta, pre_alpha = evaluate_portfolio(portfolio_df, pre_st_que)
    cur_np, cur_sharpe_ratio, cur_sortino_ratio, cur_mdd_money, cur_mdd_ratio, cur_max_hm, cur_occupy_ratio, cur_vol, cur_beta, cur_alpha = evaluate_portfolio(portfolio_df, cur_st_que)
    combined_np, combined_sharpe_ratio, combined_sortino_ratio, combined_mdd_money, combined_mdd_ratio, combined_max_hm, combined_occupy_ratio, combined_vol, combined_beta, combined_alpha = evaluate_portfolio(portfolio_df, pre_st_que + cur_st_que)
    
    return combined_mdd_ratio < min(pre_mdd_ratio, cur_mdd_ratio) and (combined_occupy_ratio > max(pre_occupy_ratio, cur_occupy_ratio))


def get_preformance(profit_series, holding_money_series):
    
    return round(profit_series.sum() / 10000, 2), \
            round(get_MDD(profit_series) / 10000, 2), round(holding_money_series.cummax().iloc[-1] / 10000, 2), \
            round(holding_money_series[holding_money_series > 0].mean() / 10000, 2), \
        round((holding_money_series.max() - holding_money_series).sum() / (holding_money_series.max() * len(holding_money_series)), 2)

def portfolio_corr(portfolio_df, col1, col2):
    merge_df = pd.merge(portfolio_df[['Date', col1]], portfolio_df[['Date', col2]], on=['Date'], how='inner').dropna()
    merge_df = merge_df[(merge_df[col1] != 0) & (merge_df[col2] != 0)]
    return round(merge_df[col1].corr(merge_df[col2]), 2)


def get_st_profit_hm(portfolio_df, st_que):
    profit_list = [st + '_profit' for st in st_que]
    hm_list = [st + '_holding_money' for st in st_que]

    profit_series = portfolio_df[profit_list].sum(axis=1)
    holding_money_series = portfolio_df[hm_list].sum(axis=1)

    return profit_series, holding_money_series.dropna()

def get_st_info(portfolio_df, st_que):
    return get_preformance(*get_st_profit_hm(portfolio_df, st_que))

def merge_action(merge_condition, portfolio_report, portfolio_cols):
    portfolio_df = portfolio_report.copy()
    portfolio_deque = deque()
    for portfolio_col in portfolio_cols:
        portfolio_deque.append([portfolio_col])

    counter = 0
    while portfolio_deque:
        pre_st_que = portfolio_deque.popleft()
        if len(pre_st_que) > 1 or ( pre_st_que == [portfolio_cols[0]] and counter > 0 ) :
            portfolio_deque.append(pre_st_que)
            break
        
        portfolio_non_list = []
        while portfolio_deque:
            cur_st_que = portfolio_deque.popleft()
            #print(pre_st_que, cur_st_que)
            if merge_condition(portfolio_df, pre_st_que, cur_st_que):
                pre_st_que = pre_st_que + cur_st_que
            else:
                portfolio_non_list.append(cur_st_que)
                
        for portfolio_col in portfolio_non_list:
            portfolio_deque.append(portfolio_col)
        portfolio_deque.append(pre_st_que)
        counter += 1
        #print('portfolio_deque', portfolio_deque)
    
    return portfolio_deque

def merge_st_profit_hm(portfolio_df, st_que, merge_portfolio_cols):

    profit_list = [st + '_profit' for st in st_que]
    hm_list = [st + '_holding_money' for st in st_que]

    st_name = st_que[0]
    if len(st_que) > 1:
        for st in st_que[1:]:
            st_name += '|' + st 
    profit_name = st_name + '_profit'
    hm_name = st_name + '_holding_money'
    merge_portfolio_cols.append(st_name)

    profit_series = portfolio_df[profit_list].sum(axis=1)
    holding_money_series = portfolio_df[hm_list].sum(axis=1)

    portfolio_df[profit_name] = profit_series
    portfolio_df[hm_name] = holding_money_series


def portfolio_merge(merge_condition, portfolio_df, portfolio_trading_record_df, portfolio_cols):
    
    merge_portfolio_df = portfolio_df.copy()
    portfolio_deque = merge_action(merge_condition, merge_portfolio_df, portfolio_cols)
        
    merge_portfolio_cols = []
    for st_que in portfolio_deque:
        merge_st_profit_hm(merge_portfolio_df, st_que, merge_portfolio_cols)
        
    merge_portfolio_trading_record_df = portfolio_trading_record_df.copy()

    for cols in merge_portfolio_cols:
        for col in cols.split('|'):
            merge_portfolio_trading_record_df.loc[merge_portfolio_trading_record_df.strategy == col, 'strategy'] = cols
 
    return merge_portfolio_df, merge_portfolio_trading_record_df, merge_portfolio_cols

def get_portfolio_merge(portfolio_report, portfolio_cols):

    portfolio_df = portfolio_report.copy()
    
    strategy_merge_list = []
    
    for pre_index in range(len(portfolio_cols)-1):
        pre_profit_series = portfolio_df[f'{portfolio_cols[pre_index]}_profit']
        pre_holding_money_series = portfolio_df[f'{portfolio_cols[pre_index]}_holding_money']
        pre_np, pre_mdd, pre_max_hm, pre_mean_hm, pre_occupy_ratio = get_preformance(pre_profit_series, pre_holding_money_series.dropna())
       
        strategy_merge_list.append( [portfolio_cols[pre_index]] + [pre_np, pre_mdd, pre_max_hm, pre_mean_hm, pre_occupy_ratio, np.nan, np.nan])
        
        for cur_index in range(pre_index + 1, len(portfolio_cols)):
            cur_profit_series = portfolio_df[f'{portfolio_cols[cur_index]}_profit']
            cur_holding_money_series = portfolio_df[f'{portfolio_cols[cur_index]}_holding_money']
            cur_np, cur_mdd, cur_max_hm, cur_mean_hm, cur_occupy_ratio = get_preformance(cur_profit_series, cur_holding_money_series.dropna())

            profit_series = portfolio_df[[f'{portfolio_cols[pre_index]}_profit', f'{portfolio_cols[cur_index]}_profit']].sum(axis=1)
            holding_money_series = portfolio_df[[f'{portfolio_cols[pre_index]}_holding_money', f'{portfolio_cols[cur_index]}_holding_money']].sum(axis=1)
            combined_np, combined_mdd, combined_max_hm, combined_mean_hm, combined_occupy_ratio = get_preformance(profit_series, holding_money_series.dropna())
            
            
            profit_corr = portfolio_corr(portfolio_df, f'{portfolio_cols[pre_index]}_profit', f'{portfolio_cols[cur_index]}_profit')
            hm_corr = portfolio_corr(portfolio_df, f'{portfolio_cols[pre_index]}_holding_money', f'{portfolio_cols[cur_index]}_holding_money')

            strategy_merge_list.append([' + ' + portfolio_cols[cur_index]] + [combined_np, combined_mdd, combined_max_hm, combined_mean_hm, combined_occupy_ratio, profit_corr, hm_corr])
        strategy_merge_list.append([])
       
    strategy_merge_list.append([portfolio_cols[-1]] + [*get_preformance(portfolio_df[f'{portfolio_cols[-1]}_profit'], portfolio_df[f'{portfolio_cols[-1]}_holding_money'].dropna()), np.nan, np.nan])
    
    portfolio_merge_table = pd.DataFrame(strategy_merge_list, columns=['策略', '淨利(萬)', '區間最大回撤(萬)', '最大持倉金額(萬)', '平均持倉金額(萬)', '佔有比率', '損益相關性', '持倉金額相關性'])
    portfolio_merge_table.to_csv(report_path + '策略嵌合.csv', index = 0, encoding = 'big5')

    return portfolio_merge_table


