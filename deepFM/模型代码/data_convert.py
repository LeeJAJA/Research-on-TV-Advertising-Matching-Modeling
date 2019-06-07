import numpy as np
import pandas as pd
import utilsMCM

ad_data_path = "C:\\Users\\zhang\\Desktop\\pycharm\\mcm\\out3.txt"
user_data_path = "C:\\Users\\zhang\\Desktop\\pycharm\\mcm\\user_data_new.txt"

user_structure = ['user_id', 'user_age', 'user_gender', 'user_area', 'user_status', 'user_education',
                  'user_consuption', 'user_device', 'user_work', 'user_connection', 'user_behavior']
##id 3 18 1 13 2 8 5 15 12 14
ad_structure = ['3', '18', '1', '13', '2', '8', '5', '15', '12', '14']
ad_data = pd.read_csv(ad_data_path, delimiter = " ")
user_data = pd.read_csv(user_data_path, delimiter = ",", names = user_structure)

dict = []

print(ad_data.iloc[0], ad_data[:100])
print(user_data[:100])

