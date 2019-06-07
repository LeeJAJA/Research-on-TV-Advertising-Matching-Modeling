import numpy as np
import pandas as pd
import utilsMCM

ad_data_path = "C:\\Users\\zhang\\Desktop\\pycharm\\mcm\\out3.txt"
user_data_path = "C:\\Users\\zhang\\Desktop\\pycharm\\mcm\\user_data_new.txt"

#ad_structure = ['ad_id', 'ad_time', 'ad_count_id', 'goods_id', 'goods_type', 'goods_class', 'size' ]
user_structure = [' ', 'user_id', 'user_age', 'user_gender', 'user_area', 'user_status', 'user_education',
                  'user_consuption', 'user_device', 'user_work', 'user_connection', 'user_behavior']

ad_data = pd.read_csv(ad_data_path, delimiter = " ")
user_data = pd.read_csv(user_data_path, delimiter = ",", header = 0, names = user_structure)

dict = [3, 18, 1,  13,   2,   8,   5,  15,  12,  14]

for i in range(len(dict)):
    user_data[str(i)] = ad_data[str(dict[i])]

#user_data['target'] = 1


behavior = user_data.pop('user_behavior')
area = user_data.pop('user_area')
work = user_data.pop('user_work')
status = user_data.pop('user_status')


print(ad_data)
print(user_data[:9000])
print(user_data[:10])
user_data[:9000].to_csv('train2.csv', index = None)

