from parse import parse_args
from load_data import Data
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import heapq

args = parse_args()
#print(args.dataset)

data = Data(args)
sorted_id, belong, rate, usersorted_id, userbelong, userrate = data.plot_pics()
Ks = eval(args.Ks)

# dataset=['addressa','globe','gowalla','ml_10m','yelp2018']
# for s in dataset:
#     print(s)
#     args.dataset=s
#     data = Data(args)


#     sorted_id, belong, rate, usersorted_id, userbelong, userrate = data.plot_pics()

#     Ks = eval(args.Ks)


#     pop_user=np.array([len(item) for item in data.train_user_list.values()])
#     pop_item=np.array([len(item) for item in data.train_item_list.values()])
#     distinct_pop_user=len(set(pop_user))
#     distinct_pop_item=len(set(pop_item))
#     print("distinct_pop_user:",distinct_pop_user)
#     print("distinct_pop_item:",distinct_pop_item)

#     #  matplotlib.axes.Axes.hist() 方法的接口
#     plt.clf()
#     n, bins, patches = plt.hist(x=pop_user, bins=args.pop_partition_user, color='#0504aa',
#                                 alpha=0.7, rwidth=0.85)
#     plt.grid(axis='y', alpha=0.75)
#     plt.xlabel('Popularity')
#     plt.ylabel('Frequency')
#     plt.title('User popularity histogram, '+args.dataset)
#     maxfreq = n.max()
#     #print(n)
#     plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
#     plt.savefig('user_'+args.dataset+'.png')


#     plt.clf()
#     n, bins, patches = plt.hist(x=pop_item, bins=args.pop_partition_item, color='#0504aa',
#                                 alpha=0.7, rwidth=0.85)
#     plt.grid(axis='y', alpha=0.75)
#     plt.xlabel('Popularity')
#     plt.ylabel('Frequency')
#     plt.title('Item popularity histogram, '+args.dataset)
#     maxfreq = n.max()
#     #print(n)
#     plt.ylim(ymax=maxfreq+10)

#     plt.savefig('item_'+args.dataset+'.png')




BATCH_SIZE = args.batch_size
ITEM_NUM = data.n_items
USER_NUM = data.n_users

points = [10, 50, 100, 200, 500]