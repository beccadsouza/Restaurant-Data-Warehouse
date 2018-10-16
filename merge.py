import pandas as pd
import numpy as np
df_customer = pd.read_csv("customer")
df_orders = pd.read_csv("orders")
df_items = pd.read_csv("items")
df_time = pd.read_csv("time")
print(pd.merge(df_customer,df_orders,on='cust_id',how="left"))
df_merge_item_orders = pd.merge(df_orders,df_items,on='item_id',how='left')
temp2 = df_merge_item_orders[['time_id','name','quantity']]
table = pd.pivot_table(temp2, values='quantity', index=['time_id'],columns=['name'], aggfunc=np.sum)
