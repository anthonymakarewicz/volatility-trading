import os
import warnings 
import py7zr
import pandas as pd

warnings.filterwarnings('ignore')

def extract_options(path):
    dir_list = os.listdir(path)
    options_data = pd.DataFrame()

    for zipfile_ in dir_list:
        with py7zr.SevenZipFile(path+zipfile_, 'r') as archive:
            archive.extractall()
            all_files = archive.getnames()

            for file_name in all_files:
                monthly_data = pd.read_csv(file_name)
                options_data = pd.concat([options_data, monthly_data], ignore_index=True)
                os.remove(file_name)
    
    return options_data


def extract_option_chain(path, expiry_date):
    options = extract_options(path)
    options = options[options[" [EXPIRE_DATE]"].str.contains(expiry_date)]
    option_chain = options[[" [C_BID]", " [C_ASK]"," [STRIKE]", " [P_BID]", " [P_ASK]"]]
    option_chain.columns = ["bid_call", "ask_call", "strike", "bid_put", "ask_put"]
    option_chain = option_chain.apply(pd.to_numeric, errors='coerce')

    return option_chain