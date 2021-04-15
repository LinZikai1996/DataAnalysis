users = ['user1','user2','user3','user4','user5']

items = ['itemA','itemB','itemC','itemD','itemE']

# 用户购买记录
datasets = [
    [1,0,1,1,0],
    [1,0,0,1,1],
    [1,0,1,0,0],
    [0,1,0,1,1],
    [1,1,1,0,1]
]

import pandas as pd

df = pd.DataFrame(datasets, columns=items, index=users)

from sklearn.metrics.pairwise import pairwise_distances
user_similar = 1 - pairwise_distances(df.values, metric="jaccard")
user_similar = pd.DataFrame(user_similar, columns=users, index=users)

# 物品相关性
item_similar = 1 - pairwise_distances(df.T.values, metric="jaccard")
item_similar = pd.DataFrame(item_similar, columns=items, index=items)

topN_users = {}
# 为每个用户找到最相似的用户

for i in user_similar.index:
    # 去出每一行数据，删除自己
    data = user_similar.loc[i].drop([i])
    # 大到小排序
    data_sort = data.sort_values(ascending = False)
    topN_users[i] = list(data_sort.index[:2])


import numpy as np
rs_results = {}

for user, users_sim in topN_users.items():
    rs_result = set()
    
    for user_sim in users_sim:
        rs_result = rs_result.union(set(df.loc[user_sim].replace(0, np.nan).dropna().index))
    
    rs_result -= set(df.loc[user_sim].replace(0, np.nan).dropna().index)

    rs_results[user] = rs_result

print(rs_results)