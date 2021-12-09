'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import collections

class Data(object):
    def __init__(self, path, batch_size, neg_sample, args):
        self.path = path
        self.batch_size = batch_size
        self.neg_sample = neg_sample

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []
        
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    if len(items)!=0:
                        self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    if len(items) != 0:
                        self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        # self.n_items += 1
        # self.n_users += 1
        self.print_statistics()
        
        self.train_items, self.test_set = {}, {}
        self.test_item_list = collections.defaultdict(list)
        self.test_user_list = collections.defaultdict(list)
        self.train_user_list = collections.defaultdict(list)
        self.train_item_list = collections.defaultdict(list)
        self.valid_user_list = collections.defaultdict(list)
        self.valid_item_list = collections.defaultdict(list)
        self.test_id_user_list = collections.defaultdict(list)
        self.test_id_item_list = collections.defaultdict(list)
        
        train_file = self.path + '/train.txt'
        valid_file = self.path + '/valid.txt'
        test_file = self.path + '/test.txt'
        test_id_file=self.path + '/test_id.txt'
        self.evalsets={}
        with open(train_file) as f:
            for line in f.readlines():
                line = line.strip('\n').split(' ')
                if len(line) == 0:
                    continue
                line = [int(i) for i in line]
                user = line[0]
                items = line[1:]
                if (len(items)==0):
                    continue
                self.train_user_list[user] = items
                for item in items:
                    self.train_item_list[item].append(user)
                self.n_users = max(self.n_users, user)
                self.n_items = max(self.n_items, max(items))
                self.n_train += len(items)
        print('train')
        print(max(self.train_user_list.keys()) + 1)
        self.n_train_users = self.n_users
        self.n_train_items = self.n_items
        

        

        with open(test_file) as f:
            for line in f.readlines():
                line = line.strip('\n').split(' ')
                if len(line) == 0:
                    continue
                line = [int(i) for i in line]
                user = line[0]
                items = line[1:]
                if len(items) == 0:
                    continue
                self.test_user_list[user] = items
                for item in items:
                    self.test_item_list[item].append(user)
                self.n_users = max(self.n_users, user)
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)
        self.evalsets['test']= self.test_user_list
        self.test_users = set(self.test_user_list.keys())

        print('test')
        print(max(self.test_user_list.keys()) + 1)
        if ".new" in args.dataset:
            with open(valid_file) as f:
                for line in f.readlines():
                    line = line.strip('\n').split(' ')
                    if len(line) == 0:
                        continue
                    line = [int(i) for i in line]
                    user = line[0]
                    items = line[1:]
                    if len(items) == 0:
                        continue
                    self.valid_user_list[user] = items
                    #self.valid_items.update(set(items))
                    for item in items:
                        self.valid_item_list[item].append(user)
                    self.n_users = max(self.n_users, user)
                    self.n_items = max(self.n_items, max(items))
                    #self.n_valid += len(items)
            self.valid_users = set(self.valid_user_list.keys())
            self.evalsets['valid']= self.valid_user_list

            print('valid')
            print(max(self.valid_user_list.keys()) + 1)


            with open(test_id_file) as f:
                for line in f.readlines():
                    line = line.strip('\n').split(' ')
                    if len(line) == 0:
                        continue
                    line = [int(i) for i in line]
                    user = line[0]
                    items = line[1:]
                    if len(items) == 0:
                        continue
                    self.test_id_user_list[user] = items
                    for item in items:
                        self.test_id_item_list[item].append(user)
                    self.n_users = max(self.n_users, user)
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
            self.evalsets['test_id']= self.test_id_user_list


            print('test_id')
            print(max(self.test_id_user_list.keys()) + 1)

        self.n_users = self.n_users + 1
        self.n_items = self.n_items + 1
        print(self.n_users)
        print(self.n_items)
            
        pop_user={key:len(value) for key,value in self.train_user_list.items()}
        pop_item={key:len(value) for key,value in self.train_item_list.items()}
        self.pop_item=pop_item
        sorted_pop_user=list(set(list(pop_user.values())))
        sorted_pop_item=list(set(list(pop_item.values())))
        sorted_pop_user.sort()
        sorted_pop_item.sort()
        #print(sorted_pop_user)
        #print(sorted_pop_item)
        self.user_pop_num=len(sorted_pop_user)
        self.item_pop_num=len(sorted_pop_item)
        user_idx={}
        item_idx={}
        for i, item in enumerate(sorted_pop_user):
            user_idx[item]=i
        for i, item in enumerate(sorted_pop_item):
            item_idx[item]=i
        self.user_pop_idx={}
        self.item_pop_idx={}
        for i in range(self.n_users):
            if i in pop_user:
                self.user_pop_idx[i]=user_idx[pop_user[i]]
            else:
                self.user_pop_idx[i]=0
        for i in range(self.n_items):
            if i in pop_item:
                self.item_pop_idx[i]=item_idx[pop_item[i]]
            else:
                self.item_pop_idx[i]=0

        def get_degrees(dataset, n_node):
            degrees = np.array(
                [len(dataset[u]) if u in dataset else 0 for u in range(n_node)],
                dtype=np.int32)
            return degrees
        
        def invert_dict(d, sort=False):
            inverse = {}
            for key in d:
                for value in d[key]:
                    if value not in inverse:
                        inverse[value] = [key]
                    else:
                        inverse[value].append(key)
            return inverse

        
        self.train=self.train_user_list
        self.train_inverse=invert_dict(self.train)

        self.users = list(range(self.n_users))
        self.items = list(range(self.n_items))
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_pop = sp.dok_matrix((self.user_pop_num, self.item_pop_num), dtype=np.float32)


        for u,items in self.train_user_list.items():
            for i in items:
                self.R[u,i]=1
                self.R_pop[self.user_pop_idx[u],self.item_pop_idx[i]]=1



        self.u_degrees = get_degrees(self.train, self.n_users)
        self.i_degrees = get_degrees(self.train_inverse, self.n_items)
 
        self.train = [
            self.train_user_list[u] if u in self.train_user_list else []
            for u in range(self.n_users)
        ]
        self.train_inverse = [
            self.train_inverse[i] if i in self.train_inverse else []
            for i in range(self.n_items)
        ]
        self.u_interacts = []
        self.i_interacts = []
        for u, items in enumerate(self.train):
            for i in items:
                self.u_interacts.append(u)
                self.i_interacts.append(i)
        self.u_interacts = np.array(self.u_interacts, dtype=np.int32)
        self.i_interacts = np.array(self.i_interacts, dtype=np.int32)
        self.n_interact = self.u_interacts.shape[0]
            
    def get_adj_mat(self,is_pop=False):

        name="pop" if is_pop else ""
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat'+name+'.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat'+name+'.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat'+name+'.npz')
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat'+name+'.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)
            return adj_mat, norm_adj_mat, mean_adj_mat,pre_adj_mat
        except Exception:
            print('calculating adj matrix...')
                    
        adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat(is_pop)
        

        # try:
        #     pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
        # except Exception:
        adj_mat=adj_mat
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat_inv)
        print('generate pre adjacency matrix.')
        pre_adj_mat = norm_adj.tocsr()
        sp.save_npz(self.path + '/s_pre_adj_mat'+name+'.npz', norm_adj)
        sp.save_npz(self.path + '/s_adj_mat'+name+'.npz', adj_mat)
        sp.save_npz(self.path + '/s_norm_adj_mat'+name+'.npz', norm_adj_mat)
        sp.save_npz(self.path + '/s_mean_adj_mat'+name+'.npz', mean_adj_mat)
            
        return adj_mat, norm_adj_mat, mean_adj_mat,pre_adj_mat

    def create_adj_mat(self,is_pop=False):
        t1 = time()
        if is_pop==False:
            adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.R.tolil()
            # prevent memory from overflowing
            for i in range(5):
                adj_mat[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5), self.n_users:] =\
                R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)]
                adj_mat[self.n_users:,int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)] =\
                R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)].T
            adj_mat = adj_mat.todok()
        else:
            adj_mat = sp.dok_matrix((self.user_pop_num + self.item_pop_num, self.user_pop_num + self.item_pop_num), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.R_pop.tolil()
            # prevent memory from overflowing
            for i in range(5):
                adj_mat[int(self.user_pop_num*i/5.0):int(self.user_pop_num*(i+1.0)/5), self.user_pop_num:] =\
                R[int(self.user_pop_num*i/5.0):int(self.user_pop_num*(i+1.0)/5)]
                adj_mat[self.user_pop_num:,int(self.user_pop_num*i/5.0):int(self.user_pop_num*(i+1.0)/5)] =\
                R[int(self.user_pop_num*i/5.0):int(self.user_pop_num*(i+1.0)/5)].T
            adj_mat = adj_mat.todok()

        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        
        t2 = time()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp
        
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)
        
        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
        
    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]


        def sample_pos_items_for_u(u, num):
            pos_items = list(self.train_user_list[u])
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_user_list[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_user_list[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    
    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.test_set.keys()) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                union_set = list(self.test_set[u])
                if u in self.train_items.keys():
                    union_set += self.train_items[u]
                if neg_id not in union_set and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items
    
        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    
    
    
    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)
    
    


        return split_uids, split_state

    def check(self):
        for uid in range(20):
            if self.train_items.__contains__(uid) and self.test_set.__contains__(uid):
                if len(set(self.train_items[uid]) & set(self.test_set[uid]))!=0:
                    print(uid)
    

    def sample_cause(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.users, self.batch_size)
        else:
            users = [rd.choice(self.users) for _ in range(self.batch_size)]

        pos_items, neg_items = [], []

        for user in users:
            if self.train_user_list[user] == []:
                pos_items.append(0)
            else:
                item=rd.choice(self.train_user_list[user])
                weight=0.1*self.n_train/len(self.pop_item)/self.pop_item[item]
                if weight>=1:
                    weight=0#/self.pop_item[item]
                rad=rd.random()
                if rad<weight:
                    item=item+self.n_items
                pos_items.append(item)
            while True:
                neg_item = rd.choice(self.items)
                if neg_item not in self.train_user_list[user]:
                    neg_items.append(neg_item)
                    break

        for i in range(len(users)):
            if pos_items[i] >= self.n_items:
                neg_items[i] += self.n_items
        
        all_items=pos_items+neg_items
        ctrl_items = [i+self.n_items if i<self.n_items else i-self.n_items for i in all_items]

        return users, pos_items, neg_items, all_items, ctrl_items

    

    def sample_infonce_test(self,user_pop_idx,item_pop_idx,method="infonce"):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_user_list.keys(), self.batch_size)
        else:
            users = [rd.choice(self.test_user_list.keys()) for _ in range(self.batch_size)]
        
        if method=="infonce":
            neg_sample=self.neg_sample
        else:
            neg_sample=1
        
        users_pop = []

        pos_items, neg_items = [], []
        pos_items_pop,neg_items_pop = [], []

        for user in users:
            if self.test_user_list[user] == []:
                pos_items.append(0)
            else:
                pos_items.append(rd.choice(self.test_user_list[user]))
            cnt=0

            #neg_items=rd.sample(self.train_sus_list, 64)
            while True:
                neg_item = rd.choice(self.items)
                if neg_item not in self.train_user_list[user] and neg_item not in self.valid_user_list[user] and neg_item not in self.test_user_list[user] and neg_item not in self.test_id_user_list:
                    neg_items.append(neg_item)
                    cnt+=1
                    if cnt==neg_sample:
                        break

        return users, pos_items, neg_items
    


    def sample_infonce(self,user_pop_idx,item_pop_idx):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.users, self.batch_size)
        else:
            users = [rd.choice(self.users) for _ in range(self.batch_size)]

        
        users_pop = []

        pos_items, neg_items = [], []
        pos_items_pop,neg_items_pop = [], []

        for user in users:
            if user in user_pop_idx:
                users_pop.append(user_pop_idx[user])
            else:
                users_pop.append(0)
            if self.train_user_list[user] == []:
                pos_items.append(0)
            else:
                pos_items.append(rd.choice(self.train_user_list[user]))
            cnt=0

            #neg_items=rd.sample(self.train_sus_list, 64)
            while True:
                neg_item = rd.choice(self.items)
                if neg_item not in self.train_user_list[user]:
                    neg_items.append(neg_item)
                    cnt+=1
                    if cnt==self.neg_sample:
                        break

        for item in pos_items:
            if item in item_pop_idx:
                pos_items_pop.append(item_pop_idx[item])
            else:
                pos_items_pop.append(0)
        
        for item in neg_items:
            if item in item_pop_idx:
                neg_items_pop.append(item_pop_idx[item])
            else:
                neg_items_pop.append(0)

        return users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop


    def sample_infonce_inbatch(self,user_pop_idx,item_pop_idx):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.users, self.batch_size)
        else:
            users = [rd.choice(self.users) for _ in range(self.batch_size)]

        
        users_pop = []

        pos_items, neg_items = [], []
        pos_items_pop,neg_items_pop = [], []

        for user in users:
            if user in user_pop_idx:
                users_pop.append(user_pop_idx[user])
            else:
                users_pop.append(0)
            if self.train_user_list[user] == []:
                pos_items.append(0)
            else:
                pos_items.append(rd.choice(self.train_user_list[user]))

        for item in pos_items:
            if item in item_pop_idx:
                pos_items_pop.append(item_pop_idx[item])
            else:
                pos_items_pop.append(0)
        

        pos_items=np.array(pos_items)
        pos_items_pop=np.array(pos_items_pop)

        neg_items=np.tile(pos_items,(pos_items.shape[0],1))
        neg_items_pop=np.tile(pos_items_pop,(pos_items_pop.shape[0],1))

        neg_items=np.reshape(neg_items[~np.eye(neg_items.shape[0],dtype=bool)],-1)
        neg_items_pop=np.reshape(neg_items_pop[~np.eye(neg_items_pop.shape[0],dtype=bool)],-1)

        return users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop
