import glob
import configparser
import json
import datetime
import os
config = configparser.ConfigParser()
config.read("config.ini")

class Config():
    def __init__(self):
        config = configparser.ConfigParser()
        config.read("config.ini")
        nowtime_str = datetime.datetime.now().strftime("%Y%m%d")

        self.xq_output_path = config['path']['xq_output_path']
        self.xq_output_paths = glob.glob(f"{self.xq_output_path}{nowtime_str}*")
        self.signal_print_path = config["path"]["signal_print_path"]
        self.signal_print_paths = glob.glob(f"{self.signal_print_path}*")
        self.signal_print_backup_path = config["path"]["signal_print_backup_path"]
        self.strategy_status_path = config["path"]["strategy_status_path"]
        self.strategy_names = [strategy_name[:-4] for strategy_name in os.listdir(path=self.strategy_status_path)] 
        self.deal_log = config["file"]["deal_log"]
        self.order_log = config['file']['order_log']

        with open(config['file']['strategy_info_file'], "r") as f:
            self.strategy_info = json.load(f)

        # determine if application is a script file or frozen exe
        #if getattr(sys, 'frozen', False):
        #    application_path = os.path.dirname(sys.executable)
        #elif __file__:
        #    application_path = os.path.dirname(__file__)

        #config_path = os.path.join(application_path, 'config.ini')
        #print('config_path = ', config_path)
                
        self.strategy_config = configparser.ConfigParser()
        self.strategy_config.read(config['file']['strategy_config'], encoding='utf-8')


class EnvConfig:
    # database
    DB_HOST = config["database"]["DB_HOST"]
    DB_PORT = int(config["database"]["DB_PORT"])
    DB_USER = config["database"]["DB_USER"]
    DB_PASSWORD = config["database"]["DB_PASSWORD"]
    DB_DATABASE = config["database"]["DB_DATABASE"]