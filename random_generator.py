import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

num = 200000

df = pd.DataFrame(columns=['广告行业','投放时段','电视台','用户年龄段','用户性别','教育水平','所在行业','消费水平'])

s = pd.Series(["酒类","家用电器","食品","饮料","邮电通讯"])
example_weights = [22, 13, 18, 19, 6]
df['广告行业'] = s.sample(n=num, weights=example_weights, replace=True).reset_index(drop=True)

s = pd.Series(["0:00-2:00","2:00-4:00","4:00-6:00","6:00-8:00","8:00-10:00","10:00-12:00","12:00-14:00","14:00-16:00","16:00-18:00","18:00-20:00","20:00-22:00","22:00-0:00"])
example_weights = [3, 3, 5, 2, 6,4,1,2,4,3,12,14]
df['投放时段'] =s.sample(n=num, weights=example_weights, replace=True).reset_index(drop=True)

s = pd.Series(["湖南卫视","浙江卫视","CCTV","东方卫视"])
example_weights = [1,2,8,3]
df['电视台'] =s.sample(n=num, weights=example_weights, replace=True).reset_index(drop=True)

s = pd.Series(["0-10","10-20","20-30","30+"])
example_weights = [5,10,12,40]
df['用户年龄段'] =s.sample(n=num, weights=example_weights, replace=True).reset_index(drop=True)

s = pd.Series(["男","女"])
example_weights = [94,106]
df['用户性别'] =s.sample(n=num, weights=example_weights, replace=True).reset_index(drop=True)

s = pd.Series(["高中及以下","大专","本科及以上"])
example_weights = [76,112,285]
df['教育水平'] =s.sample(n=num, weights=example_weights, replace=True).reset_index(drop=True)

s = pd.Series(["教育","金融保险","社会公共管理","IT电子通信","医药卫生","住宿旅游"])
example_weights = [91,134,120,109,114,95]
df['所在行业'] =s.sample(n=num, weights=example_weights, replace=True).reset_index(drop=True)

s = pd.Series(["低","中" ,"高"])
example_weights = [93,95,120]
df['消费水平'] =s.sample(n=num, weights=example_weights, replace=True).reset_index(drop=True)

df=df.reset_index(drop=True)
df.to_csv('sample.csv')
df