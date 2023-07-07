import glob
import configparser
import json
import datetime
import os

class BaseConfig():
    def __init__(self):
        config = configparser.ConfigParser()
        config.read("strategy_config.ini", encoding='utf-8')
        nowtime_str = datetime.datetime.now().strftime("%Y%m%d")

        self.portfolio_interday_path = config['path']['portfolio_interday_path']
        self.portfolio_intraday_path = config['path']['portfolio_intraday_path']

Config = BaseConfig()
