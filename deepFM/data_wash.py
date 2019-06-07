import numpy as np
import pandas as pd
import utilsMCM


user_data_path = "C:\\Users\\zhang\\Desktop\\pycharm\\mcm\\user_data_new.txt"

user_structure = ['user_id', 'user_age', 'user_gender', 'user_area', 'user_status', 'user_education',
                  'user_consuption', 'user_device', 'user_work', 'user_connection', 'user_behavior']

user_data = pd.read_csv(user_data_path, delimiter = ",", names = user_structure)


print(user_data[:100])

behavior = user_data.pop('user_gender')
behavior = user_data.pop('user_age')
area = user_data.pop('user_area')
work = user_data.pop('user_work')
status = user_data.pop('user_status')
behavior = user_data.pop('user_education')
area = user_data.pop('user_consuption')
work = user_data.pop('user_device')
work = user_data.pop('user_connection')

print(user_data[10000:11000])
user_data[10000:11000].to_csv("train_f_test.txt")

