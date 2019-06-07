import numpy as np
import pandas as pd
import utilsMCM

ad_data_path = "E:\\testA\\st_f.txt"
user_data_path = "C:\\Users\\zhang\\Desktop\\pycharm\\mcm\\user_data_new.txt"

ad_structure = ['ad_id', 'ad_time', 'ad_count_id', 'goods_id', 'goods_type', 'goods_class', 'size' ]
user_structure = ['user_id', 'user_age', 'user_gender', 'user_area', 'user_status', 'user_education',
                  'user_consuption', 'user_device', 'user_work', 'user_connection', 'user_behavior']

ad_data = pd.read_csv(ad_data_path, delimiter = "\t", nrows = 1000, names = ad_structure)
user_data = pd.read_csv(user_data_path, delimiter = ",", nrows = 11000, names = user_structure)

print(ad_data)
print(user_data)
print("successfully load !")

"""
dict = []
dict.append(ad_data.iloc[0]['goods_type'])
for i in range(len(ad_data)):
    if utilsMCM.Exist(dict, ad_data.iloc[i]['goods_type']) == False:
        dict.append(ad_data.iloc[i]['goods_type'])
        print(dict)

for i in range(len(dict)):
    user_data[str(dict[i])] = 0

k =0

for i in range(len(user_data)):
    behavior = user_data.iloc[i]['user_behavior'].split(',')
    for j in range(len(behavior)):
        try:
            adID = int(behavior[j])
        except:
            continue
        for o in range(len(ad_data)):
            #print(adID, int(ad_data.iloc[o]['ad_id']), k, i)
            if int(ad_data.iloc[o]['ad_id']) - adID == 0:
                user_data.iloc[i][str(ad_data.iloc[o]['goods_type'])] += 1
                print(adID, int(ad_data.iloc[o]['ad_id']), k, i)
                k += 1
"""



"""
d = 0
user_data_new = pd.DataFrame(columns = ['userID', 'userBehavior'])
for i in range(len(user_data)):
    v = pd.DataFrame({'userID':[user_data.iloc[i]['user_id']], 'userBehavior':[user_data.iloc[i]['user_behavior']]})
    user_data_new = user_data_new.append(v)
print(user_data_new,len(user_data_new))

user_data_new.to_csv("user_data_new2.txt")
"""

d = 0
user_data_new = pd.DataFrame(columns = user_structure)
for i in range(len(user_data)):
    if i > 10000:
        v = user_data.iloc[i]
        user_data_new = user_data_new.append(v)
        print(i)
print(user_data_new, len(user_data_new))

user_data_new.to_csv("ad_data_new4.txt")

