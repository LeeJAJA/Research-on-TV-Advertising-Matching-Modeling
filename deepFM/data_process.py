import numpy as np
import pandas as pd
import utilsMCM
from pandas.compat import StringIO, BytesIO


def search(user, feature_dict, value):
    gender = str(user['user_gender'])
    education = str(user['user_education'])
    consuption = str(user['user_consuption'])
    device = str(user['user_device'])
    work = str(user['user_work'])

    for i in range(len(feature_dict)):
        if feature_dict[i]['gender'] == gender and feature_dict[i]['education'] == education and feature_dict[i]['consuption'] == consuption and feature_dict[i]['device'] == device and feature_dict[i]['work'] == work:
            feature_dict[i]['3'] += value['3']
            feature_dict[i]['18'] += value['18']
            feature_dict[i]['1'] += value['1']
            feature_dict[i]['13'] += value['13']
            feature_dict[i]['2'] += value['2']
            feature_dict[i]['8'] += value['8']
            feature_dict[i]['5'] += value['5']
            feature_dict[i]['15'] += value['15']
            feature_dict[i]['12'] += value['12']
            feature_dict[i]['14'] += value['14']
    return feature_dict



link_path = "C:\\Users\\zhang\\Desktop\\pycharm\\mcm\\out3.txt"

user_data_path = "C:\\Users\\zhang\\Desktop\\pycharm\\mcm\\user_data_new.txt"

user_structure = ['user_id', 'user_age', 'user_gender', 'user_area', 'user_status', 'user_education',
                  'user_consuption', 'user_device', 'user_work', 'user_connection', 'user_behavior']

user_data = pd.read_csv(user_data_path, delimiter = ",", names = user_structure)
link = pd.read_csv(link_path, delimiter = ' ')

dict = [3, 18, 1,  13,   2,   8,   5,  15,  12,  14]

print("data_loading done !")
print("clear starts ！")

print(user_data.describe().astype(np.int64).T)

dict_age = []
dict_age.append(user_data.iloc[0]['user_age'])

dict_gender = []
dict_gender.append(user_data.iloc[0]['user_gender'])

dict_education = []
dict_education.append(user_data.iloc[0]['user_education'])

dict_consuption = []
dict_consuption.append(user_data.iloc[0]['user_consuption'])

dict_device = []
dict_device.append(user_data.iloc[0]['user_device'])

dict_work = []
dict_work.append(user_data.iloc[0]['user_work'])

dict_connection = []
dict_connection.append(user_data.iloc[0]['user_connection'])

for i in range(len(user_data)):
    if utilsMCM.Exist(dict_age, user_data.iloc[i]['user_age']) == False:
        dict_age.append(user_data.iloc[i]['user_age'])

    if utilsMCM.Exist(dict_gender, user_data.iloc[i]['user_gender']) == False:
        dict_gender.append(user_data.iloc[i]['user_gender'])

    if utilsMCM.Exist(dict_education, user_data.iloc[i]['user_education']) == False:
        dict_education.append(user_data.iloc[i]['user_education'])

    if utilsMCM.Exist(dict_consuption, user_data.iloc[i]['user_consuption']) == False:
        dict_consuption.append(user_data.iloc[i]['user_consuption'])

    if utilsMCM.Exist(dict_device, user_data.iloc[i]['user_device']) == False:
        dict_device.append(user_data.iloc[i]['user_device'])

    if utilsMCM.Exist(dict_work, user_data.iloc[i]['user_work']) == False:
        dict_work.append(user_data.iloc[i]['user_work'])

    if utilsMCM.Exist(dict_connection, user_data.iloc[i]['user_connection']) == False:
        dict_connection.append(user_data.iloc[i]['user_connection'])

print("gender:", len(dict_gender), dict_gender)
print("education", len(dict_education), dict_education)
print("consuption", len(dict_consuption), dict_consuption)
print("device", len(dict_device), dict_device)
print("work", len(dict_work), dict_work)

User_conbination = []

for i in range(len(dict_gender)):
    for j in range(len(dict_education)):
        for k in range(len(dict_consuption)):
            for l in range(len(dict_device)):
                for m in range(len(dict_work)):
                    User_conbination.append({'gender':str(dict_gender[i]), 'education':str(dict_education[j]),
                                                 'consuption':str(dict_consuption[k]), 'device':str(dict_device[l]),
                                                 'work':str(dict_work[m]), '3':0, '18':0, '1':0,  '13':0, '2':0,'8':0,
                                             '5':0, '15':0, '12':0, '14':0})


for i in range(len(link)):
    User_conbination = search(user_data.iloc[i], User_conbination, link.iloc[i])

for i in range(len(User_conbination)):
    print(User_conbination[i])
print(len(User_conbination))




