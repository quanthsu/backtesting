from backtesting.strategy import Strategy
from backtesting.report import Get_strategy_report, Check_strategy_list, Compute_corr
from tcri_report import get_tcri_compare
import glob
from datetime import datetime, timedelta, date
from tqdm import tqdm
import pandas as pd
import os
import configparser

config = configparser.ConfigParser()
config.read('strategy_config.ini')

strategy_path = config['base']['path']
report_path = config['base']['report_path']
report_plot_path = config['base']['report_plot_path']
tcri_risk_path = config['base']['tcri_risk_path']
file_type = config['base']['file_type']
strategy_report_paths = glob.glob(strategy_path + '*' + file_type)
intraday_start_day = datetime.strptime(config['base']['intraday_start_day'], '%Y-%m-%d')
year_days = int(config['base']['year_days'])
transaction_cost = float(config['base']['transaction_cost'])

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

mtime_diff = timedelta(weeks=4)



if __name__ == '__main__':
  
    print('TCRI Compare ...')
    get_tcri_compare()

    leaderboard_list = []
    print('Generate report ...')
    for strategy_report_path in tqdm(strategy_report_paths):
        
        try:    
            strategy_report_list = Get_strategy_report(strategy_report_path)
            leaderboard_list.append(strategy_report_list)

        except:
            print(strategy_report_path, ' Fail ! ')

    leaderboard_df = pd.DataFrame(leaderboard_list)
    
    leaderboard_df.columns = ['策略名稱', '回測資料範圍', '回測年數', '平均持倉天數', '獲利因子', '總交易次數',
    '總交易次數(一年平均)', '勝率%', '最大投入報酬率%', '最大持倉金額(萬)', '最大區間虧損率%', '風險報酬比',
    '最大連續虧損月份', '虧損年份', f'每月至少{intraday_tradingday_count_per_month_threshold}交易日有訊號', '平均每月交易日', 'A\n(X=總收/總損)\nA=(X-1)/(X+1)', 'B\n(Y=均正報酬/均負報酬)\nB=(Y-1)/(Y+1)*勝率', 'C\n(Z=平均單筆賺賠)\nC=Z/MDD', 'A+B+C']
    leaderboard_df.to_excel(report_path + "豐神榜.xlsx", sheet_name='豐神榜績效(XQ)', index = 0)  

    qualified_leaderboard_df = Check_strategy_list(leaderboard_df)

    qualified_leaderboard_df.to_excel(report_path + "豐神榜_合格.xlsx", sheet_name='豐神榜績效(XQ)', index = 0)  


    print('Calculate correlation ...')
    Compute_corr(qualified_leaderboard_df['策略名稱'], '_合格')
    Compute_corr(leaderboard_df['策略名稱'], '')

    print('Generate report plot ...')
    for strategy_report_path in tqdm(strategy_report_paths):
        #if datetime.now() - datetime.fromtimestamp(os.path.getmtime(strategy_report_path)) >= mtime_diff:
        #    continue
        #try:
            strategy = Strategy(strategy_report_path)
            strategy.get_plotly_html()
        #except:
        #    print(strategy_report_path, ' Fail ! ')

    print('Complete ! ')

    os.system("pause")
