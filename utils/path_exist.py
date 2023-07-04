import os

def check_folder_path_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
