import pandas as pd
# import numpy as np

df_customer = pd.read_csv("customer")
df_orders = pd.read_csv("orders")
df_items = pd.read_csv("items")
df_time = pd.read_csv("time")


print(pd.merge(df_customer,df_orders,on='cust_id',how="left"))
