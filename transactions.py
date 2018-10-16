import pandas as pd
import numpy as np
df_customer = pd.read_csv("customer")
df_orders = pd.read_csv("orders")
df_items = pd.read_csv("items")
df_time = pd.read_csv("time")

df_merge_item_orders = pd.merge(df_orders,df_items,on='item_id',how='left')
df_trans = df_merge_item_orders[['time_id','name','quantity']]
df = pd.pivot_table(df_trans, values='quantity', index=['time_id'],columns=['name'], aggfunc=np.sum)

trans = []
for i in range(len(df)):
    l = []
    for x in df:
        if df.iloc[i,:][x]>0: l.append(x)
    trans.append(l)

for x in trans:print(x)

