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

class Analytics(object):
    
    def DD(self, navRecords_dd = None):
        
        if navRecords_dd is None:
            navRecords_dd = self.navRecords.copy()
        
        maxDD_ratio = 0
        maxDD_num = 0
        peak_price = 0        
        
        maxDDList_ratio = []
        DDList_ratio = []
        maxDDList_num = []
        DDList_num = []
        for date in navRecords_dd.index:
            
            curPrice = navRecords_dd.loc[date, 'nav']
            #print(date, curPrice)
            if curPrice > peak_price:
                peak_price = curPrice
                
                maxDDList_ratio.append([date, maxDDList_ratio[-1][1] if maxDDList_ratio else 0])
                DDList_ratio.append([date, 0])
                
                maxDDList_num.append([date, maxDDList_num[-1][1] if maxDDList_num else 0])
                DDList_num.append([date, 0])

            else:
                DD_ratio = curPrice / peak_price - 1
                DD_num = curPrice - peak_price
                
                if DD_ratio < maxDD_ratio:
                    maxDD_ratio = DD_ratio
                
                if DD_num < maxDD_num:
                    maxDD_num = DD_num
                
                maxDDList_ratio.append([date, maxDD_ratio])
                DDList_ratio.append([date, DD_ratio])
                
                maxDDList_num.append([date, maxDD_num])
                DDList_num.append([date, DD_num])

        maxDDFrame_ratio = pd.DataFrame(maxDDList_ratio, columns = ['date', 'maxDD_ratio'])
        maxDDFrame_ratio.set_index('date', inplace = True)
        DDFrame_ratio = pd.DataFrame(DDList_ratio, columns = ['date', 'DD_ratio'])
        DDFrame_ratio.set_index('date', inplace = True)
        
        DDFrame_ratio = pd.concat([DDFrame_ratio, maxDDFrame_ratio], axis = 1)
        
        maxDDFrame_num = pd.DataFrame(maxDDList_num, columns = ['date', 'maxDD_num'])
        maxDDFrame_num.set_index('date', inplace = True)
        DDFrame_num = pd.DataFrame(DDList_num, columns = ['date', 'DD_num'])
        DDFrame_num.set_index('date', inplace = True)
        
        DDFrame_num = pd.concat([DDFrame_num, maxDDFrame_num], axis = 1)
        
        DDFrame = pd.concat([DDFrame_ratio, DDFrame_num], axis = 1)

        return DDFrame

    def _get_daily_holding_nums(self, tradingRecords_nums = None):
        
        if tradingRecords_nums is None:
            tradingRecords_nums = self.tradingRecords.copy()
        
        numsList = []
        for date in self.date_list:
            in_datetime = datetime(date.year, date.month, date.day, 13, 30)

            #dailyHoldingRecrods = tradingRecords_nums[(tradingRecords_nums['inDate'].dt.date <= date) & (tradingRecords_nums['outDate'].dt.date >= date)]
            trading_period = ((tradingRecords_nums['inDate'] <= in_datetime) & (tradingRecords_nums['outDate'] > in_datetime )) \
                    | ((tradingRecords_nums['inDate'] <= in_datetime) & pd.isnull(tradingRecords_nums['outDate']))
            dailyHoldingRecrods = tradingRecords_nums[trading_period]
            
            #numsList.append([date, dailyHoldingRecrods.shape[0]])
            numsList.append([date, len(dailyHoldingRecrods['ticker'].unique())])
            
        numsDF = pd.DataFrame(numsList, columns = ['date', 'HoldingNums'])
        numsDF.set_index('date', inplace = True)
        
        return numsDF


    '''
    def _get_daily_holding_money(self, tradingRecords = None):
        if tradingRecords is None:
            tradingRecords = self.tradingRecords.copy()
        
        moneyList = []
        for date in self.date_list:
            in_datetime = datetime(date.year, date.month, date.day, 13, 30) 
            dailyHoldingRecrods = tradingRecords[(tradingRecords['inDate'] <= in_datetime) & (tradingRecords['outDate'] >= in_datetime)]
            
            moneyList.append([date, (dailyHoldingRecrods['inPrice'] * dailyHoldingRecrods['trading_number']).sum() * 1000])
        
        moneyDF = pd.DataFrame(moneyList, columns=['date', 'HoldingMoney'])
        moneyDF.set_index('date', inplace = True)
        return moneyDF
    '''
    
    def _get_daily_holding_money(self, tradingRecords = None, tcri_limit = False):
        if tradingRecords is None:
            tradingRecords = self.tradingRecords.copy()
        
        moneyList = []
        for date in self.date_list:
            in_datetime = datetime(date.year, date.month, date.day, 13, 30) 
            trading_period = ((tradingRecords['inDate'] <= in_datetime) & (tradingRecords['outDate'] > in_datetime )) \
                    | ((tradingRecords['inDate'] <= in_datetime) & pd.isnull(tradingRecords['outDate']))
            dailyHoldingRecrods = tradingRecords[trading_period]

            moneyList.append([date, (dailyHoldingRecrods['inPrice'] * dailyHoldingRecrods['trading_number']).sum() * 1000])
        
        moneyDF = pd.DataFrame(moneyList, columns=['date', 'HoldingMoney'])
        moneyDF.set_index('date', inplace = True)
        return moneyDF

    def _get_daily_trading_nums(self, tradingRecords_nums = None):
        
        if tradingRecords_nums is None:
            tradingRecords_nums = self.tradingRecords.copy()
        
        numsList = []
        for date in self.date_list:
            
            dailyHoldingRecrods = tradingRecords_nums[tradingRecords_nums['inDate'].dt.date == date]
            #numsList.append([date, dailyHoldingRecrods.shape[0]])
            numsList.append([date, len(dailyHoldingRecrods['ticker'].unique())])
    
        numsDF = pd.DataFrame(numsList, columns = ['date', 'TradingNums'])
        numsDF.set_index('date', inplace = True)
        
        return numsDF
    
    def _strategy_daily_returns(self, navFrame_ret = None):
        
        if navFrame_ret is None:
            navFrame_ret = self.navRecords.copy()
            
        lnRetFrame = np.log(navFrame_ret / navFrame_ret.shift(1)).fillna(0)
        lnRetFrame.columns = ['lnReturns']
        
        return lnRetFrame
        
    def _get_maxContinousNumber(self, navFrame, direction):
        
        if direction == "Profit":
            
            directionNum = 0
            ContinuousBeginPoint = navFrame.iloc[0]['nav']
            for i in range(1, navFrame.shape[0]):
                
                curNav = navFrame.iloc[i]['nav']
                lastNav = navFrame.iloc[i - 1]['nav']
                
                if curNav > lastNav:
                    if (curNav - ContinuousBeginPoint) > directionNum:
                        directionNum = curNav - ContinuousBeginPoint
            
                else:
                    ContinuousBeginPoint = curNav
            
            
        elif direction == "Loss":
        
            directionNum = 0
            ContinuousBeginPoint = navFrame.iloc[0]['nav']
            for i in range(1, navFrame.shape[0]):
                
                curNav = navFrame.iloc[i]['nav']
                lastNav = navFrame.iloc[i - 1]['nav']
                
                if curNav < lastNav:
                    if (curNav - ContinuousBeginPoint) < directionNum:
                        directionNum = curNav - ContinuousBeginPoint
            
                else:
                    ContinuousBeginPoint = curNav
                
        return directionNum
    
    def _get_maxPeriodNumber(self, navFrame, direction):
        
        if direction == "Profit":
            
            directionNum = 0
            upperBound = navFrame.iloc[0]['nav']
            lowerBound = navFrame.iloc[0]['nav']
            
            for i in range(1, navFrame.shape[0]):
                
                curNav = navFrame.iloc[i]['nav']
                
                if (curNav < lowerBound) | (i == navFrame.shape[0] - 1):
                    
                    if (upperBound - lowerBound) > directionNum:
                        directionNum = upperBound - lowerBound
                        
                    upperBound = curNav
                    lowerBound = curNav
                
                elif curNav > upperBound:
                    upperBound = curNav
                    
                    
        elif direction == "Loss":
        
            directionNum = 0
            upperBound = navFrame.iloc[0]['nav']
            lowerBound = navFrame.iloc[0]['nav']
            
            for i in range(1, navFrame.shape[0]):
                
                curNav = navFrame.iloc[i]['nav']
                
                if (curNav > upperBound) | (i == navFrame.shape[0] - 1):
            
                    if (lowerBound - upperBound) < directionNum:
                        directionNum = lowerBound - upperBound
                        
                    upperBound = curNav
                    lowerBound = curNav
                
                elif curNav < lowerBound:
                    lowerBound = curNav    
        
        
        return directionNum
    
    def stat(self, show = False):
        
        tradingRecords_stat = self.tradingRecords.copy()
        navRecords_stat = self.navRecords.copy()
        
        #限制資料在回測區間
        #tradingRecords_stat = tradingRecords_stat[(tradingRecords_stat['inDate'] >= self.BeginDate) & (tradingRecords_stat['outDate'] <= self.endDate)]
        #navRecords_stat = navRecords_stat[(navRecords_stat.index >= self.BeginDate) & (navRecords_stat.index <= self.endDate)]
        #navRecords_stat = navRecords_stat / navRecords_stat.iloc[0]  #rebased到回測區間第一天
        
        netNavRecords = navRecords_stat - 1
        DDFrame = self.DD(navRecords_stat)
        dailyHoldingNums = self._get_daily_holding_nums(tradingRecords_stat)
        dailyRetFrame = self._strategy_daily_returns(navRecords_stat)
        monthlyRetFrame = dailyRetFrame.resample('M').sum()
        yearlyRetFrame = dailyRetFrame.resample('Y').sum()
        
        MonthlyNavFrame = np.exp(monthlyRetFrame).cumprod()
        YearlyNavFrame = np.exp(yearlyRetFrame).cumprod()
        
        MonthlyNavFrame.columns = ['nav']
        YearlyNavFrame.columns = ['nav']        
        
        monthlyRetFrame.index = [index.strftime('%Y-%m') for index in monthlyRetFrame.index]
        yearlyRetFrame.index = [index.strftime('%Y') for index in yearlyRetFrame.index]
        
        
        #總表(筆)
        ###總報酬率(數值)
        totRet_num = navRecords_stat.iloc[-1]['nav'] - navRecords_stat.iloc[0]['nav']
        ###總報酬率(百分比)
        totRet_ratio = navRecords_stat.iloc[-1]['nav'] / navRecords_stat.iloc[0]['nav'] - 1
        ###總毛利
        grossProfit = tradingRecords_stat[tradingRecords_stat['returns'] > 0]['returns'].sum()
        ###總毛損
        grossLoss = tradingRecords_stat[tradingRecords_stat['returns'] <= 0]['returns'].sum()
        ###總淨利
        netProfit = grossProfit - abs(grossLoss)
        ###平均獲利(百分比)
        avgGrossProfit = grossProfit / tradingRecords_stat[tradingRecords_stat['returns'] > 0]['returns'].shape[0]
        ###平均虧損(百分比)
        avgGrossLoss = grossLoss / tradingRecords_stat[tradingRecords_stat['returns'] <= 0]['returns'].shape[0]
        ###平均淨利(百分比)
        avgNetProfit = netProfit / tradingRecords_stat.shape[0]
        ###獲利因子
        profitFactor = grossProfit / abs(grossLoss)
        ###平均獲利虧損比(賺賠比)
        profitLossRatio = abs(avgGrossProfit / avgGrossLoss)
        ###最大回檔(數值)
        maxDD_num = abs(DDFrame.iloc[-1]['maxDD_num'])
        ###最大回檔(百分比)
        maxDD_ratio = abs(DDFrame.iloc[-1]['maxDD_ratio'])
        ###單日歷史VaR(95%)
        VaR95_daily = np.exp(dailyRetFrame['lnReturns'].quantile(0.05)) - 1
        ###單月歷史VaR(95%)
        VaR95_monthly = np.exp(monthlyRetFrame['lnReturns'].quantile(0.05)) - 1
        ###最大持有個股數
        maxHoldingNums = dailyHoldingNums['HoldingNums'].max()
        ###CAGR & MAR
        years = (navRecords_stat.index[-1] - navRecords_stat.index[0]).days / 365
        cagr = (navRecords_stat.iloc[-1]['nav'] / navRecords_stat.iloc[0]['nav']) ** (1 / years) - 1
        mar = cagr / maxDD_ratio
        ###總交易次數
        totTradingCount = tradingRecords_stat.shape[0]
        ###獲利次數
        winTradingCount = tradingRecords_stat[tradingRecords_stat['returns'] > 0].shape[0]
        ###虧損次數
        lossTradingCount = tradingRecords_stat[tradingRecords_stat['returns'] <= 0].shape[0]
        ###勝率
        winRatio = winTradingCount / totTradingCount
        ###最大單筆交易獲利
        maxProfit_oneTrade = tradingRecords_stat['returns'].max()
        ###最大單筆交易虧損
        minProfit_oneTrade = tradingRecords_stat['returns'].min()
        ###最大連續獲利
        maxContinuousProfit = self._get_maxContinousNumber(navRecords_stat, "Profit")
        ###最大連續虧損
        maxContinuousLoss = self._get_maxContinousNumber(navRecords_stat, "Loss")
        ###最大區間獲利
        maxPeriodProfit = self._get_maxPeriodNumber(navRecords_stat, "Profit")
        ###最大區間虧損
        maxPeriodLoss = self._get_maxPeriodNumber(navRecords_stat, "Loss")
        

        #日
        ###獲利比例(日)
        winRate_daily = dailyRetFrame[dailyRetFrame['lnReturns'] > 0].shape[0] / dailyRetFrame.shape[0]
        ###交易日數
        totTradingDays = dailyRetFrame.shape[0]
        ###獲利日數
        totWinDays = dailyRetFrame[dailyRetFrame['lnReturns'] > 0].shape[0]
        ###虧損日數
        totLossDays = dailyRetFrame[dailyRetFrame['lnReturns'] <= 0].shape[0]
        ###平均毛利(百分比)
        avgGrossProfit_daily = np.exp(dailyRetFrame[dailyRetFrame['lnReturns'] > 0]['lnReturns'].mean()) - 1
        ###平均毛損(百分比)
        avgGrossLoss_daily = np.exp(dailyRetFrame[dailyRetFrame['lnReturns'] <= 0]['lnReturns'].mean()) - 1
        ###平均淨利(百分比)
        avgNetProfit_daily = np.exp(dailyRetFrame['lnReturns'].mean()) - 1
        ###最大單日獲利
        maxDailyProfit = np.exp(dailyRetFrame['lnReturns'].max()) - 1
        ###最大單日虧損
        maxDailyLoss = np.exp(dailyRetFrame['lnReturns'].min()) - 1
        ###最大連續獲利
        maxContinuousProfit_daily = maxContinuousProfit
        ###最大連續虧損
        maxContinuousLoss_daily = maxContinuousLoss
        ###最大區間獲利
        maxPeriodProfit_daily = maxPeriodProfit
        ###最大區間虧損
        maxPeriodLoss_daily = maxPeriodLoss
        
        
        #月
        ###獲利比例(月)
        winRate_monthly = monthlyRetFrame[monthlyRetFrame['lnReturns'] > 0].shape[0] / monthlyRetFrame.shape[0]
        ###交易月數
        totTradingMonths = monthlyRetFrame.shape[0]
        ###獲利月數
        totWinMonths = monthlyRetFrame[monthlyRetFrame['lnReturns'] > 0].shape[0]
        ###虧損月數
        totLossMonths = monthlyRetFrame[monthlyRetFrame['lnReturns'] <= 0].shape[0]
        ###平均毛利(百分比)
        avgGrossProfit_monthly = np.exp(monthlyRetFrame[monthlyRetFrame['lnReturns'] > 0]['lnReturns'].mean()) - 1
        ###平均毛損(百分比)
        avgGrossLoss_monthly = np.exp(monthlyRetFrame[monthlyRetFrame['lnReturns'] <= 0]['lnReturns'].mean()) - 1
        ###平均淨利(百分比)
        avgNetProfit_monthly = np.exp(monthlyRetFrame['lnReturns'].mean()) - 1
        ###最大單月獲利
        maxMonthlyProfit = np.exp(monthlyRetFrame['lnReturns'].max()) - 1
        ###最大單月虧損
        maxMonthlyLoss = np.exp(monthlyRetFrame['lnReturns'].min()) - 1
        ###最大連續獲利
        maxContinuousProfit_monthly = self._get_maxContinousNumber(MonthlyNavFrame, "Profit")
        ###最大連續虧損
        maxContinuousLoss_monthly = self._get_maxContinousNumber(MonthlyNavFrame, "Loss")
        ###最大區間獲利
        maxPeriodProfit_monthly = self._get_maxPeriodNumber(MonthlyNavFrame, "Profit")
        ###最大區間虧損
        maxPeriodLoss_monthly = self._get_maxPeriodNumber(MonthlyNavFrame, "Loss")
        
        
        #年
        ###獲利比例(年)
        winRate_yearly = yearlyRetFrame[yearlyRetFrame['lnReturns'] > 0].shape[0] / yearlyRetFrame.shape[0]
        ###交易年數
        totTradingYears = yearlyRetFrame.shape[0]
        ###獲利年數
        totWinYears = yearlyRetFrame[yearlyRetFrame['lnReturns'] > 0].shape[0]
        ###虧損年數
        totLossYears = yearlyRetFrame[yearlyRetFrame['lnReturns'] <= 0].shape[0]
        ###平均毛利(百分比)
        avgGrossProfit_yearly = np.exp(yearlyRetFrame[yearlyRetFrame['lnReturns'] > 0]['lnReturns'].mean()) - 1
        ###平均毛損(百分比)
        avgGrossLoss_yearly = np.exp(yearlyRetFrame[yearlyRetFrame['lnReturns'] <= 0]['lnReturns'].mean()) - 1
        ###平均淨利(百分比)
        avgNetProfit_yearly = np.exp(yearlyRetFrame['lnReturns'].mean()) - 1
        ###最大單年獲利
        maxYearlyProfit = np.exp(yearlyRetFrame['lnReturns'].max()) - 1
        ###最大單年虧損
        maxYearlyLoss = np.exp(yearlyRetFrame['lnReturns'].min()) - 1
        ###最大連續獲利
        maxContinuousProfit_yearly = self._get_maxContinousNumber(YearlyNavFrame, "Profit")
        ###最大連續虧損
        maxContinuousLoss_yearly = self._get_maxContinousNumber(YearlyNavFrame, "Loss")
        ###最大區間獲利
        maxPeriodProfit_yearly = self._get_maxPeriodNumber(YearlyNavFrame, "Profit")
        ###最大區間虧損
        maxPeriodLoss_yearly = self._get_maxPeriodNumber(YearlyNavFrame, "Loss")
                
        #數字統計(展示用)
        statDF_show = pd.DataFrame()
        
        #總表(筆)
        statDF_show.insert(0, "總表", ['總報酬(數值)', '總報酬(百分比)', '總毛利(萬)', '總毛損(萬)', '總淨利(萬)', '獲利因子', '平均獲利虧損比(賺賠比)', '最大回檔(數值)', '最大回檔(百分比)', '95%VaR_單日(百分比)', '95%VaR_單月(百分比)', 'CAGR', 'MAR'], allow_duplicates=True)
        statDF_show.insert(1, "總表", ['{:.2f}'.format(totRet_num), '{:.2%}'.format(totRet_ratio), '{:.2f}'.format(grossProfit/10000), '{:.2f}'.format(grossLoss/10000), '{:.2f}'.format(netProfit/10000), '{:.2f}'.format(profitFactor), '{:.2f}'.format(profitLossRatio), '{:.2f}'.format(maxDD_num), '{:.2%}'.format(maxDD_ratio), '{:.2%}'.format(VaR95_daily), '{:.2%}'.format(VaR95_monthly), '{:.2%}'.format(cagr), '{:.2f}'.format(mar)], allow_duplicates=True)
        #筆
        statDF_show.insert(2, "筆", ['勝率', '總交易次數', '獲利次數', '虧損次數', '平均淨利(萬)_筆', '平均獲利(萬)_筆', '平均虧損(萬)_筆', '最大單筆交易獲利(萬)', '最大單筆交易虧損(萬)', '最大連續獲利(百分比)_筆', '最大連續虧損(百分比)_筆', '最大區間獲利(百分比)_筆', '最大區間虧損(百分比)_筆'], allow_duplicates=True)
        statDF_show.insert(3, "筆", ['{:.2%}'.format(winRatio), '{:.0f}'.format(totTradingCount), '{:.0f}'.format(winTradingCount), '{:.0f}'.format(lossTradingCount), '{:.2f}'.format(avgNetProfit/10000), '{:.2f}'.format(avgGrossProfit/10000), '{:.2f}'.format(avgGrossLoss/10000), '{:.2f}'.format(maxProfit_oneTrade/10000), '{:.2f}'.format(minProfit_oneTrade/10000), '{:.2f}'.format(maxContinuousProfit), '{:.2f}'.format(maxContinuousLoss), '{:.2f}'.format(maxPeriodProfit), '{:.2f}'.format(maxPeriodLoss)], allow_duplicates=True)
        #日
        statDF_show.insert(4, "日", ['日獲利比率', '交易日數', '獲利日數', '虧損日數', '平均淨利(百分比)_日', '平均獲利(百分比)_日', '平均虧損(百分比)_日', '最大單日獲利', '最大單日虧損', '最大連續獲利_日', '最大連續虧損_日', '最大區間獲利_日', '最大區間虧損_日'], allow_duplicates=True)
        statDF_show.insert(5, "日", ['{:.2%}'.format(winRate_daily), '{:.0f}'.format(totTradingDays), '{:.0f}'.format(totWinDays), '{:.0f}'.format(totLossDays), '{:.2%}'.format(avgNetProfit_daily), '{:.2%}'.format(avgGrossProfit_daily), '{:.2%}'.format(avgGrossLoss_daily), '{:.2%}'.format(maxDailyProfit), '{:.2%}'.format(maxDailyLoss), '{:.2f}'.format(maxContinuousProfit_daily), '{:.2f}'.format(maxContinuousLoss_daily), '{:.2f}'.format(maxPeriodProfit_daily), '{:.2f}'.format(maxPeriodLoss_daily)], allow_duplicates=True)
        #月
        statDF_show.insert(6, "月", ['月獲利比率', '交易月數', '獲利月數', '虧損月數', '平均淨利(百分比)_月', '平均獲利(百分比)_月', '平均虧損(百分比)_月', '最大單月獲利', '最大單月虧損', '最大連續獲利_月', '最大連續虧損_月', '最大區間獲利_月', '最大區間虧損_月'], allow_duplicates=True)
        statDF_show.insert(7, "月", ['{:.2%}'.format(winRate_monthly), '{:.0f}'.format(totTradingMonths), '{:.0f}'.format(totWinMonths), '{:.0f}'.format(totLossMonths), '{:.2%}'.format(avgNetProfit_monthly), '{:.2%}'.format(avgGrossProfit_monthly), '{:.2%}'.format(avgGrossLoss_monthly), '{:.2%}'.format(maxMonthlyProfit), '{:.2%}'.format(maxMonthlyLoss), '{:.2f}'.format(maxContinuousProfit_monthly), '{:.2f}'.format(maxContinuousLoss_monthly), '{:.2f}'.format(maxPeriodProfit_monthly), '{:.2f}'.format(maxPeriodLoss_monthly)], allow_duplicates=True)
        #年
        statDF_show.insert(8, "月", ['年獲利比率', '交易年數', '獲利年數', '虧損年數', '平均淨利(百分比)_年', '平均獲利(百分比)_年', '平均虧損(百分比)_年', '最大單年獲利', '最大單年虧損', '最大連續獲利_年', '最大連續虧損_年', '最大區間獲利_年', '最大區間虧損_年'], allow_duplicates=True)
        statDF_show.insert(9, "月", ['{:.2%}'.format(winRate_yearly), '{:.0f}'.format(totTradingYears), '{:.0f}'.format(totWinYears), '{:.0f}'.format(totLossYears), '{:.2%}'.format(avgNetProfit_yearly), '{:.2%}'.format(avgGrossProfit_yearly), '{:.2%}'.format(avgGrossLoss_yearly), '{:.2%}'.format(maxYearlyProfit), '{:.2%}'.format(maxYearlyLoss), '{:.2f}'.format(maxContinuousProfit_yearly), '{:.2f}'.format(maxContinuousLoss_yearly), '{:.2f}'.format(maxPeriodProfit_yearly), '{:.2f}'.format(maxPeriodLoss_yearly)], allow_duplicates=True)
        
        statDF_show.columns = [['總表', '總表', '筆', '筆', '日', '日', '月', '月', '年', '年'], ['項目', '數值', '項目', '數值', '項目', '數值', '項目', '數值', '項目', '數值']]
        
        
        fig_stat = go.Figure(data=[go.Table(
            
            header=dict(values=list(statDF_show.columns),
                        #fill_color='paleturquoise',
                        align='center'),
            
            cells=dict(values=[statDF_show[column] for column in statDF_show.columns],
                        #fill_color='lavender',
                        align='left'))
                                
        ])
        
        fig_stat.update_layout(width=1700, height=500)
        
        if show:
            fig_stat.show()
        
        return statDF_show, fig_stat
