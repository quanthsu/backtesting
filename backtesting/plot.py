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
import seaborn as sns
from plotly.subplots import make_subplots
from backtesting.evaluate import evaluate_portfolio
#from bs4 import BeautifulSoup
import base64
import configparser

config = configparser.ConfigParser()
config.read('strategy_config.ini')

strategy_path = config['base']['path']
report_path = config['base']['report_path']
report_plot_path = config['base']['report_plot_path']
report_plot_portfolio_path = config['base']['report_plot_portfolio_path']
trading_day_date_path = config['base']['trading_day_date_path']
twa02_path = config['base']['twa02_path']
twc02_path = config['base']['twc02_path']
file_type = config['base']['file_type']
img_path = config['base']['img_path']
year_days = int(config['base']['year_days'])
transaction_cost = float(config['base']['transaction_cost'])
risk_free_rate = float(config['base']['risk_free_rate'])
total_asset = float(config['allocation']['total_asset'])

class Plot(object):    
        
    def get_DD(self):
        ###開始畫圖
        #累積報酬率、DD組圖
        tradingRecords_plot = self.tradingRecords.copy()
        navRecords_plot = self.navRecords.copy()
        
        netNavRecords = navRecords_plot - 1
        DDFrame = self.DD(navRecords_plot)

        fig_DD = make_subplots(rows = 3, cols = 1, subplot_titles = ('策略走勢圖', 'DD走勢圖', 'MDD走勢圖'), shared_xaxes = True)
        
        fig_DD.add_trace(go.Scatter(x = netNavRecords.index, y = netNavRecords['nav'], fill='tozeroy', name = '策略走勢'), row = 1, col = 1)
        #fig_DD.add_trace(go.Scatter(x = netIndexNavFrame.index, y = netIndexNavFrame['nav'], mode = 'lines', name = '大盤走勢'), row = 1, col = 1)
        
        fig_DD.add_trace(go.Scatter(x = DDFrame.index, y = DDFrame['DD_ratio'], fill='tozeroy', name = 'DD走勢'), row = 2, col = 1)
        fig_DD.add_trace(go.Scatter(x = DDFrame.index, y = DDFrame['maxDD_ratio'], fill='tozeroy', name = 'MDD走勢'), row = 3, col = 1)
        
        
        fig_DD.update_xaxes(
                            rangeslider = {'visible' : True, 'thickness': 0.05},
                            rangeselector = {
                                        'buttons' : list([
                                            dict(count=1, label="1m", step="month", stepmode="backward"),
                                            dict(count=6, label="6m", step="month", stepmode="backward"),
                                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                                            dict(count=1, label="1y", step="year", stepmode="backward"),
                                            dict(step="all")
                                        ])
                            }
        )
        
        
        fig_DD.update_layout(
                            height = 900, 
                            width = 1700, #1000
                            title = {'text': '<b>策略走勢圖與回檔幅度<b>', 'x': 0.5, 'font': {'size': 30}}
                            )
        
        #fig_DD.show()
        return fig_DD
        
    def get_pos(self):

        tradingRecords_plot = self.tradingRecords.copy()
        navRecords_plot = self.navRecords.copy()
        
        netNavRecords = navRecords_plot - 1
        dailyHoldingNums = self._get_daily_holding_nums(tradingRecords_plot)
        dailyHoldingNums = navRecords_plot.join(dailyHoldingNums)[['HoldingNums']]
        dailyHoldingMoney = self._get_daily_holding_money(tradingRecords_plot)
        dailyHoldingMoney = navRecords_plot.join(dailyHoldingMoney)[['HoldingMoney']]
        dailyTradingNums = self._get_daily_trading_nums(tradingRecords_plot)
        dailyTradingNums = navRecords_plot.join(dailyTradingNums)[['TradingNums']]
            
        #累積報酬率、持有部位走勢圖
        fig_pos = make_subplots(rows = 4, cols = 1, subplot_titles = ('策略走勢圖', '持有檔數走勢圖', '進場檔數走勢圖', '持倉水位走勢圖'), shared_xaxes = True)
        
        fig_pos.add_trace(go.Scatter(x = netNavRecords.index, y = netNavRecords['nav'], fill='tozeroy', name = '策略走勢'), row = 1, col = 1)
        #fig_pos.add_trace(go.Scatter(x = netIndexNavFrame.index, y = netIndexNavFrame['nav'], mode = 'lines', name = '大盤走勢'), row = 1, col = 1)
        
        fig_pos.add_trace(go.Scatter(x = dailyHoldingNums.index, y = dailyHoldingNums['HoldingNums'], name = '持有檔數走勢圖'), row = 2, col = 1)
        fig_pos.add_trace(go.Scatter(x = dailyTradingNums.index, y = dailyTradingNums['TradingNums'], name = '進場檔數走勢圖'), row = 3, col = 1)
        fig_pos.add_trace(go.Scatter(x = dailyHoldingMoney.index, y = dailyHoldingMoney['HoldingMoney'], name = '持倉水位走勢圖'), row = 4, col = 1)

        fig_pos.update_xaxes(
                            rangeslider = {'visible' : True, 'thickness': 0.05},
                            rangeselector = {
                                        'buttons' : list([
                                            dict(count=1, label="1m", step="month", stepmode="backward"),
                                            dict(count=6, label="6m", step="month", stepmode="backward"),
                                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                                            dict(count=1, label="1y", step="year", stepmode="backward"),
                                            dict(step="all")
                                        ])
                            }
        )
        
        
        fig_pos.update_layout(
                            height = 700, 
                            width = 1700, #1700
                            title = {'text': '<b>策略走勢圖與持有部位走勢圖<b>', 'x': 0.5, 'font': {'size': 30}}
                            )
        
        #fig_pos.show()
        return fig_pos
        
    def get_hist(self):

        tradingRecords_plot = self.tradingRecords.copy()
        tradingRecords_plot['returns'] /= 10000
        #報酬率分布圖
        outest_prob = 0.015
        bars_num = 10
        
        returnsList = list(tradingRecords_plot['returns'])
        upperLimit = round(tradingRecords_plot['returns'].quantile(1 - outest_prob / 2), 2)
        lowerLimit = round(tradingRecords_plot['returns'].quantile(outest_prob / 2), 2)
        
        jumpRange = (upperLimit - lowerLimit) / bars_num
        
        resultDict = {}
        colorList = []
        
        resultDict[('< ' + ('{:.2f}'.format(float(lowerLimit))))] = tradingRecords_plot[tradingRecords_plot['returns'] < lowerLimit].shape[0]

        if lowerLimit <= 0:
            colorList.append('下跌')
        else:
            colorList.append('上漲')

        for i in range(bars_num):
            
            upperBound = lowerLimit + jumpRange * (i + 1)
            lowerBound = lowerLimit + jumpRange * i
            
            RangeName = ('>= ' + ('{:.2f}'.format(float(lowerBound)))) + ('  < ' + ('{:.2f}'.format(float(upperBound))))
            
            resultDict[RangeName] = tradingRecords_plot[(tradingRecords_plot['returns'] >= lowerBound) & (tradingRecords_plot['returns'] < upperBound)].shape[0]
            
            ##如果小於0的比數超過1半，那就綠色，不然就紅色
            if tradingRecords_plot[(tradingRecords_plot['returns'] >= lowerBound) & (tradingRecords_plot['returns'] < 0)].shape[0] > 0.5 * resultDict[RangeName]:
                colorList.append('下跌')
            else:
                colorList.append('上漲')
        
        resultDict[('>= ' + ('{:.2f}'.format(float(upperLimit))))] = tradingRecords_plot[tradingRecords_plot['returns'] >= upperLimit].shape[0]
        
        if upperLimit <= 0:
            colorList.append('下跌')
        else:
            colorList.append('上漲')
        
        resultFrame = pd.DataFrame(resultDict.items(), columns = ['range', 'count'])
        resultFrame = resultFrame.assign(color = colorList)
        resultFrame.set_index('range', inplace = True)

        fig_hist = px.bar(resultFrame, x = resultFrame.index, y= resultFrame['count'], color = 'color', color_discrete_map = {'下跌': 'green', '上漲': 'red'}, title='時間加權報酬率走勢圖')
        
        
        fig_hist.update_layout(
                            showlegend=False,
                            height = 700, 
                            width = 1700, #1700
                            title = {'text': '<b>報酬分布圖(萬)<b>', 'x': 0.5, 'font': {'size': 30}},
                            xaxis_tickangle=-60,
                            )
        
        #fig_hist.show()
        return fig_hist
        
    def get_freq(self):
        tradingRecords_plot = self.tradingRecords.copy()
        navRecords_plot = self.navRecords.copy()
        
        netNavRecords = navRecords_plot - 1
        
        dailyRetFrame = self._strategy_daily_returns(navRecords_plot)
        monthlyRetFrame = dailyRetFrame.resample('M').sum()
        yearlyRetFrame = dailyRetFrame.resample('Y').sum()
                
        netMonthlyNavFrame = np.exp(monthlyRetFrame).cumprod() - 1
        netYearlyNavFrame = np.exp(yearlyRetFrame).cumprod() - 1
        
        netMonthlyNavFrame.columns = ['nav']
        netYearlyNavFrame.columns = ['nav']        
        
        #日、月、年平均報酬率走勢圖
        colors = {
                    '上漲': 'red',
                    '下跌': 'green'
        }
        
        dailyRetFrame = dailyRetFrame.assign(color = [('下跌' if ret < 0 else '上漲') for ret in dailyRetFrame['lnReturns']])
        monthlyRetFrame = monthlyRetFrame.assign(color = [('下跌' if ret < 0 else '上漲') for ret in monthlyRetFrame['lnReturns']])
        yearlyRetFrame = yearlyRetFrame.assign(color = [('下跌' if ret < 0 else '上漲') for ret in yearlyRetFrame['lnReturns']])
        
        #為了讓年資料的圖與時間軸對上，需要將日期設為年度第一天
        yearlyRetFrame.index = [datetime(index.year, 1, 1) for index in yearlyRetFrame.index]
        netYearlyNavFrame.index = [datetime(index.year, 1, 1) for index in netYearlyNavFrame.index]
        #netYearlyIndexNavFrame.index = [datetime(index.year, 1, 1) for index in netYearlyIndexNavFrame.index]
        
        fig_freq = make_subplots(rows = 3, cols = 2, subplot_titles = ('日平均報酬率', '時間加權報酬率_日', '月平均報酬率', '時間加權報酬率_月', '年平均報酬率', '時間加權報酬率_年'))
        
        
        #日平均報酬率
        for col in sorted(set(dailyRetFrame['color'])):
            
            tmpDF = dailyRetFrame[dailyRetFrame['color'] == col]
            fig_freq.add_trace(go.Bar(x = tmpDF.index, y = tmpDF['lnReturns'], marker_color = colors[col]), row = 1, col = 1)
        
        
        #月平均報酬率
        for col in sorted(set(monthlyRetFrame['color'])):
            
            tmpDF = monthlyRetFrame[monthlyRetFrame['color'] == col]
            fig_freq.add_trace(go.Bar(x = tmpDF.index, y = tmpDF['lnReturns'], marker_color = colors[col]), row = 2, col = 1)
        
            
        #年平均報酬率
        for col in sorted(set(yearlyRetFrame['color'])):
            
            tmpDF = yearlyRetFrame[yearlyRetFrame['color'] == col]
            fig_freq.add_trace(go.Bar(x = tmpDF.index, y = tmpDF['lnReturns'], marker_color = colors[col]), row = 3, col = 1)
        
        
        #時間加權報酬率走勢圖_日
        fig_freq.add_trace(go.Scatter(x = netNavRecords.index, y = netNavRecords['nav'], fill='tozeroy', name = '策略走勢'), row = 1, col = 2)
        #fig_freq.add_trace(go.Scatter(x = netIndexNavFrame.index, y = netIndexNavFrame['nav'], mode = 'lines', name = '大盤走勢'), row = 1, col = 2)
        
        #時間加權報酬率走勢圖_月
        fig_freq.add_trace(go.Scatter(x = netMonthlyNavFrame.index, y = netMonthlyNavFrame['nav'], fill='tozeroy', name = '策略走勢'), row = 2, col = 2)
        #fig_freq.add_trace(go.Scatter(x = netMonthlyIndexNavFrame.index, y = netMonthlyIndexNavFrame['nav'], mode = 'lines', name = '大盤走勢'), row = 2, col = 2)
        
        #時間加權報酬率走勢圖_年  
        fig_freq.add_trace(go.Scatter(x = netYearlyNavFrame.index, y = netYearlyNavFrame['nav'], fill='tozeroy', name = '策略走勢'), row = 3, col = 2)
        #fig_freq.add_trace(go.Scatter(x = netYearlyIndexNavFrame.index, y = netYearlyIndexNavFrame['nav'], mode = 'lines', name = '大盤走勢'), row = 3, col = 2)

        fig_freq.update_xaxes(
                            rangeslider = {'visible' : True, 'thickness': 0.05},
                            rangeselector = {
                                        'buttons' : list([
                                            dict(count=1, label="1m", step="month", stepmode="backward"),
                                            dict(count=6, label="6m", step="month", stepmode="backward"),
                                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                                            dict(count=1, label="1y", step="year", stepmode="backward"),
                                            dict(step="all")
                                        ])
                            }
        )
        
        fig_freq.update_layout(
                            showlegend=False,
                            height = 1000, 
                            width = 1700, #1700
                            title = {'text': '<b>日、月、年平均報酬率與時間加權報酬率走勢圖<b>', 'x': 0.5, 'font': {'size': 30}}
                            )
        
        #fig_freq.show()
        
        return fig_freq

    def show_plotly_html(self):
        
        fig_DD = self.get_DD()
        fig_pos = self.get_pos_portfolio()
        fig_hist = self.get_hist()
        fig_freq = self.get_freq() 
        statDF_show, fig_stat = self.stat()

        fig_stat.show()
        fig_DD.show()
        fig_pos.show()
        fig_hist.show()
        fig_freq.show()

    def get_plotly_html(self):
        
        fig_DD = self.get_DD()
        fig_pos = self.get_pos()
        fig_hist = self.get_hist()
        fig_freq = self.get_freq() 
        statDF_show, fig_stat = self.stat()
        
        plot_path = self.report_plot_path + f'{self.name}_TCRI.html' if self.tcri_limit else self.report_plot_path + f'{self.name}.html'

        with open(plot_path, 'w') as f:
            f.write(fig_stat.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fig_DD.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fig_pos.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fig_hist.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fig_freq.to_html(full_html=False, include_plotlyjs='cdn'))


class Plot_portfolio(Plot):

    def get_portfolio_pos_fig(self):
        
        fig_pos = go.Figure()
        for portfolio_col in self.portfolio_cols:
            fig_pos.add_trace(go.Scatter(x=self.daily_report['Date'], y=self.daily_report[portfolio_col + '_profit'].cumsum(), name=portfolio_col))
        fig_pos.add_trace(go.Scatter(x =self.daily_report['Date'], y = self.daily_report[[portfolio_col + '_profit' for portfolio_col in self.portfolio_cols]].sum(axis=1).cumsum(), name = 'Portfolio', line=dict(color='royalblue', width=4, dash='dash')))

        fig_pos.update_layout(bargap=0)
        fig_pos.update_layout(barmode='stack')
        fig_pos.update_layout(
                            height = 700, 
                            width = 1700, #1700
                            title = {'text': '<b>策略(投組)累積損益圖<b>', 'x': 0.5, 'font': {'size': 30}}
                            )
        #fig_pos.show()
        return fig_pos

    
    def get_portfolio_profit_corr(self):
        twa_close_df = self.get_index_df(twa02_path, 'TWA02')
        twc_close_df = self.get_index_df(twc02_path, 'TWC02')

        portfolio_profit_cols = [portfolio_col + '_profit' for portfolio_col in self.portfolio_cols]
        profit_df = self.daily_report.copy()
        profit_df = pd.merge(profit_df, twa_close_df, on=['Date'], how='outer')
        profit_df = pd.merge(profit_df, twc_close_df, on=['Date'], how='outer')
        profit_df['Portfolio_profit'] = profit_df[portfolio_profit_cols].sum(axis=1)

        portfolio_twa = []
        for portfolio_col in self.portfolio_cols:
            strategy_report = profit_df[['Date', f'{portfolio_col}_profit']].replace(0, np.nan).dropna()
            profit_twa_df = pd.merge(strategy_report, twa_close_df, on=['Date'], how='inner')
            profit_twc_df = pd.merge(strategy_report, twc_close_df, on=['Date'], how='inner')
            portfolio_twa.append([portfolio_col, round(profit_twa_df[f'{portfolio_col}_profit'].corr(profit_twa_df['TWA02_profit']), 2),
            round(profit_twc_df[f'{portfolio_col}_profit'].corr(profit_twc_df['TWC02_profit']), 2)])

        profit_df = profit_df[portfolio_profit_cols+['Portfolio_profit', 'TWA02_profit', 'TWC02_profit']]
        profit_corr = profit_df.corr()
        
        profit_corr.index = self.portfolio_cols + ['Portfolio', '加權報酬指數', 'OTC報酬指數']
        profit_corr.columns = self.portfolio_cols + ['Portfolio', '加權報酬指數', 'OTC報酬指數']

        #return profit_corr.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)
        return profit_corr, pd.DataFrame(portfolio_twa, columns = ['Strategy', 'corr_加權報酬指數(持有部位區間)', 'corr_OTC報酬指數(持有部位區間)'])

    def get_pos_portfolio(self):
        
        tradingRecords_plot = self.tradingRecords.copy()
        navRecords_plot = self.navRecords.copy()
        
        netNavRecords = navRecords_plot - 1
        dailyHoldingNums = self._get_daily_holding_nums(tradingRecords_plot)
        dailyHoldingNums = navRecords_plot.join(dailyHoldingNums)[['HoldingNums']]

        dailyTradingNums = self._get_daily_trading_nums(tradingRecords_plot)
        dailyTradingNums = navRecords_plot.join(dailyTradingNums)[['TradingNums']]

        dailyHoldingMoney = self.daily_report[['Date', 'holding_money']]
        dailyHoldingMoney.columns = ['date', 'HoldingMoney']
        dailyHoldingMoney.set_index('date', inplace = True)
                    
        #累積報酬率、持有部位走勢圖
        fig_pos = make_subplots(rows = 4, cols = 1, subplot_titles = ('策略走勢圖', '持有檔數走勢圖', '進場檔數走勢圖', '持倉水位走勢圖'), shared_xaxes = True)
        
        fig_pos.add_trace(go.Scatter(x = netNavRecords.index, y = netNavRecords['nav'], fill='tozeroy', name = '策略走勢'), row = 1, col = 1)
        #fig_pos.add_trace(go.Scatter(x = netIndexNavFrame.index, y = netIndexNavFrame['nav'], mode = 'lines', name = '大盤走勢'), row = 1, col = 1)
        
        fig_pos.add_trace(go.Scatter(x = dailyHoldingNums.index, y = dailyHoldingNums['HoldingNums'], name = '持有檔數走勢圖'), row = 2, col = 1)
        fig_pos.add_trace(go.Scatter(x = dailyTradingNums.index, y = dailyTradingNums['TradingNums'], name = '進場檔數走勢圖'), row = 3, col = 1)
        fig_pos.add_trace(go.Scatter(x = dailyHoldingMoney.index, y = dailyHoldingMoney['HoldingMoney'], name = '持倉水位走勢圖'), row = 4, col = 1)

        fig_pos.update_xaxes(
                            rangeslider = {'visible' : True, 'thickness': 0.05},
                            rangeselector = {
                                        'buttons' : list([
                                            dict(count=1, label="1m", step="month", stepmode="backward"),
                                            dict(count=6, label="6m", step="month", stepmode="backward"),
                                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                                            dict(count=1, label="1y", step="year", stepmode="backward"),
                                            dict(step="all")
                                        ])
                            }
        )
        
        
        fig_pos.update_layout(
                            height = 700, 
                            width = 1700, #1700
                            title = {'text': '<b>策略走勢圖與持有部位走勢圖<b>', 'x': 0.5, 'font': {'size': 30}}
                            )
        
        #fig_pos.show()
        return fig_pos


    def get_portfolio_stat(self):
    
        portfolio_evaluation_table = []
        portfolio_evaluation_table.append(['Portfolio', *evaluate_portfolio(self.daily_report, self.portfolio_cols)])

        for index in range(len(self.portfolio_cols)):
            portfolio_evaluation_table.append([self.portfolio_cols[index], *evaluate_portfolio(self.daily_report, [self.portfolio_cols[index]])])
        
        portfolio_evaluation_df = pd.DataFrame(np.array(portfolio_evaluation_table).T[1:], columns = np.array(portfolio_evaluation_table).T[0], dtype = np.float)
    
        portfolio_evaluation_df = portfolio_evaluation_df.T
        portfolio_evaluation_df.columns = ['淨利(萬)', 'Sharpe Ratio', 'Sortino Ratio', '最大回撤金額(萬)', "MDD(%)", '最大持倉金額(萬)', "佔有比率(%)", '年化波動率(%)', 'Beta', 'Alpha']
        portfolio_evaluation_df['策略'] = portfolio_evaluation_df.index
        portfolio_evaluation_df = portfolio_evaluation_df[['策略', '淨利(萬)', 'Sharpe Ratio', 'Sortino Ratio', '最大回撤金額(萬)', "MDD(%)", '最大持倉金額(萬)', "佔有比率(%)", '年化波動率(%)', 'Beta', 'Alpha']]

        return portfolio_evaluation_df 

    def get_portfolio_compare(self):

        portfolio_evaluation_table = []
        portfolio_evaluation_table.append(['Portfolio', *evaluate_portfolio(self.daily_report, self.portfolio_cols)])

        for index in range(len(self.portfolio_cols)):
            portfolio_evaluation_table.append([self.portfolio_cols[index], *evaluate_portfolio(self.daily_report, self.portfolio_cols[:index] + self.portfolio_cols[index+1:])])
        
        portfolio_evaluation_df = pd.DataFrame(np.array(portfolio_evaluation_table).T[1:], columns = np.array(portfolio_evaluation_table).T[0], dtype = np.float)
    
        portfolio_compare_df = pd.DataFrame()
        for portfolio_col in self.portfolio_cols:
            portfolio_compare_df[portfolio_col] = round(portfolio_evaluation_df['Portfolio'] - portfolio_evaluation_df[portfolio_col], 2)
        
        portfolio_compare_df = portfolio_compare_df.T
        portfolio_compare_df.columns = ['淨利(萬)', 'Sharpe Ratio', 'Sortino Ratio', '最大回撤金額(萬)', "MDD(%)", '最大持倉金額(萬)', "佔有比率(%)", '年化波動率(%)', 'Beta', 'Alpha']
        portfolio_compare_df['策略'] = portfolio_compare_df.index
        portfolio_compare_df = portfolio_compare_df[['策略', '淨利(萬)', 'Sharpe Ratio', 'Sortino Ratio', '最大回撤金額(萬)', "MDD(%)", '最大持倉金額(萬)', "佔有比率(%)", '年化波動率(%)', 'Beta', 'Alpha']]

        return portfolio_compare_df
    
    
    def get_portfolio_stat_fig(self, portfolio_df, fig_name):
        
        fig_stat = go.Figure(data=[go.Table(
            
            header=dict(values=list(portfolio_df.columns),
                        #fill_color='paleturquoise',
                        align='center'),

            cells=dict(values=[portfolio_df[column] for column in portfolio_df.columns],
                        #fill_color='lavender',
                        align='center'))              
        ])
        fig_stat.update_layout(width=1700, height=400, title = {'text': fig_name, 'x': 0.5, 'font': {'size': 30}})
        
        #fig_stat.show()
        return fig_stat

    def show_plotly_html(self):
        
        fig_DD = self.get_DD()
        fig_pos = self.get_pos_portfolio()
        fig_hist = self.get_hist()
        fig_freq = self.get_freq() 
        statDF_show, fig_stat = self.stat()

        fig_stat.show()
        fig_DD.show()
        fig_pos.show()
        fig_hist.show()
        fig_freq.show()

    def get_img_tag(self, data_path, title):
        data_uri = base64.b64encode(open(data_path, 'rb').read()).decode('utf-8')
        img_tag = '<div style="text-align: center;"><h1>{}</h1><img src="data:image/png;base64,{}" class="center"></div>'.format(title, data_uri)
        return img_tag
    
    def get_index_df(self, index_path, index_name):
        def preprocess(daily_df):
            stock_ids = list(daily_df['股票代號'].astype('str'))
            date_indexes = daily_df.iloc[:,2:].columns
            date_indexes = [datetime.strptime(date_index[:8], "%Y%m%d") for date_index in date_indexes]
            daily_df = daily_df.T.iloc[2:, :]
            daily_df.columns = stock_ids
            daily_df.index = date_indexes
            return daily_df

        twa_close_df = pd.read_excel(index_path, sheet_name='close', header=4).dropna()
        twa_close_df = preprocess(twa_close_df)
        twa_close_df[f'{index_name}_profit'] = (twa_close_df[index_name] - twa_close_df[index_name].shift(1)).astype(float)
        twa_close_df['Date'] = twa_close_df.index
        return twa_close_df 

    '''
    def get_twa02(self):
        def preprocess(daily_df):
            stock_ids = list(daily_df['股票代號'].astype('str'))
            date_indexes = daily_df.iloc[:,2:].columns
            date_indexes = [datetime.strptime(date_index[:8], "%Y%m%d") for date_index in date_indexes]
            daily_df = daily_df.T.iloc[2:, :]
            daily_df.columns = stock_ids
            daily_df.index = date_indexes
            return daily_df

        twa_close_df = pd.read_excel(twa02_path, sheet_name='close', header=4).dropna()
        twa_close_df = preprocess(twa_close_df)
        twa_close_df['TWA02_profit'] = (twa_close_df['TWA02'] - twa_close_df['TWA02'].shift(1)).astype(float)
        twa_close_df['Date'] = twa_close_df.index
        return twa_close_df
    '''

    def get_plotly_html(self):
        plt.rcParams['font.sans-serif']=['Microsoft JhengHei']

        fig_DD = self.get_DD()
        fig_pos = self.get_pos_portfolio()
        fig_hist = self.get_hist()
        fig_freq = self.get_freq() 
        statDF_show, fig_stat = self.stat()

        portfolio_pos = self.get_portfolio_pos_fig()

        portfolio_stat_df = self.get_portfolio_stat()
        portfolio_compare_df = self.get_portfolio_compare()

        total_fig_stat =  self.get_portfolio_stat_fig(portfolio_stat_df, '<b>策略(投組)績效總覽<b>')
        compare_fig_stat = self.get_portfolio_stat_fig(portfolio_compare_df, f'<b>策略(投組)績效比較報表<b> <br><sup>加入該策略後對投組的影響</sup>')
        
        
        profit_corr, portfolio_corr = self.get_portfolio_profit_corr()
        plt.figure(figsize=(10, 10))
        profit_corr_map = sns.heatmap(profit_corr, annot=True, vmax=1, square=True, fmt=".2f", cmap='coolwarm')
        #profit_corr_map.figure.savefig(img_path + 'profit_corr.png', dpi=140) # bbox_inches = 'tight'
        plt.savefig(img_path + 'profit_corr.png', dpi=140) # bbox_inches = 'tight'
        plt.close()
        img_tag = self.get_img_tag(img_path + 'profit_corr.png', 'Portfolio Correlation Matrix')

        fig_portfolio_corr = go.Figure(data=[go.Table(
            
            header=dict(values=list(portfolio_corr.columns),
                        #fill_color='paleturquoise',
                        align='center'),
            
            cells=dict(values=[portfolio_corr[column] for column in portfolio_corr.columns],
                        #fill_color='lavender',
                        align='left'))           
        ])
        fig_portfolio_corr.update_layout(width=1700, height=500)
        
        with open(self.report_plot_path + f'{self.name}.html', 'w', encoding='utf8') as f:
            f.write(portfolio_pos.to_html())
            f.write(img_tag)
            f.write(fig_portfolio_corr.to_html(full_html=False, include_plotlyjs='cdn'))

            f.write(fig_stat.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fig_DD.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fig_pos.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fig_hist.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fig_freq.to_html(full_html=False, include_plotlyjs='cdn'))

            f.write(total_fig_stat.to_html())
            f.write(compare_fig_stat.to_html())
